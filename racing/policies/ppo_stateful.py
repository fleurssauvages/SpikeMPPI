from __future__ import annotations

"""State-preserving adapter for Brax PPO.

Brax's public PPO API deliberately returns inference parameters only.  Its
checkpoint helper likewise stores network/normalizer parameters, not the Adam
optimizer state.  That is ideal for deployment but loses optimizer momentum
when this project stops between curriculum phases.

This module derives a tiny compatibility wrapper from the *installed* Brax
``ppo.train`` source at runtime.  The numerical training implementation remains
Brax's own version; we only add two hooks:

* ``restore_training_state_bytes`` restores the full initialized TrainingState
  using Flax serialization, including Adam moments and cumulative env_steps.
* ``return_training_state=True`` returns the un-pmapped full TrainingState.

Deriving from the installed function rather than vendoring ~800 lines keeps the
adapter aligned with the Brax revision in the user's accelerator environment.
The source transformation is guarded and fails loudly if Brax changes the
relevant structure.
"""

import inspect
from pathlib import Path
from typing import Any, Callable


def _build_stateful_train() -> Callable[..., Any]:
    from brax.training.agents.ppo import train as brax_ppo_train
    from flax import serialization as flax_serialization

    native_train = brax_ppo_train.train
    sig = inspect.signature(native_train)
    # If a future Brax grows an equivalent API, use it directly only when both
    # hooks exist with the semantics we require.
    if {
        "restore_training_state_bytes",
        "return_training_state",
    }.issubset(sig.parameters):
        return native_train

    src = inspect.getsource(native_train)

    # Add our two optional arguments immediately before the closing signature.
    signature_anchor = "    run_evals: bool = True,\n):"
    if signature_anchor not in src:
        raise RuntimeError(
            "Unsupported Brax PPO train() signature: cannot install stateful "
            "curriculum adapter. Update racing/policies/ppo_stateful.py for "
            "the installed Brax revision."
        )
    src = src.replace(
        signature_anchor,
        "    run_evals: bool = True,\n"
        "    restore_training_state_bytes: Optional[bytes] = None,\n"
        "    return_training_state: bool = False,\n"
        "):",
        1,
    )

    # Full-state restoration must happen after Brax has constructed a state with
    # the correct optimizer/network pytree structure, and after public
    # param-only restoration so the full state wins when supplied.
    restore_anchor = "  if num_timesteps == 0:\n"
    if restore_anchor not in src:
        raise RuntimeError(
            "Unsupported Brax PPO train() body: TrainingState restore anchor "
            "was not found."
        )
    src = src.replace(
        restore_anchor,
        "  if restore_training_state_bytes is not None:\n"
        "    logging.info('Restoring full PPO TrainingState (including optimizer).')\n"
        "    training_state = _racing_flax_serialization.from_bytes(\n"
        "        training_state, restore_training_state_bytes\n"
        "    )\n"
        "  if num_timesteps == 0:\n",
        1,
    )

    # Return the complete unreplicated state after the ordinary Brax checks.
    return_anchor = "  pmap.synchronize_hosts()\n  return (make_policy, params, metrics)\n"
    if return_anchor not in src:
        raise RuntimeError(
            "Unsupported Brax PPO train() body: final return anchor was not found."
        )
    src = src.replace(
        return_anchor,
        "  pmap.synchronize_hosts()\n"
        "  if return_training_state:\n"
        "    return (make_policy, params, metrics, _unpmap(training_state))\n"
        "  return (make_policy, params, metrics)\n",
        1,
    )

    namespace = dict(vars(brax_ppo_train))
    namespace["_racing_flax_serialization"] = flax_serialization
    exec(compile(src, inspect.getsourcefile(native_train) or "<brax-ppo>", "exec"), namespace)
    patched = namespace["train"]
    patched.__name__ = "stateful_train"
    patched.__qualname__ = "stateful_train"
    patched.__doc__ = (native_train.__doc__ or "") + (
        "\n\nRacing extension: optionally restore/return the complete PPO TrainingState."
    )
    return patched


_stateful_train: Callable[..., Any] | None = None


def train(*args, **kwargs):
    """Calls the installed Brax PPO trainer with full-state continuation hooks."""
    global _stateful_train
    if _stateful_train is None:
        _stateful_train = _build_stateful_train()
    return _stateful_train(*args, **kwargs)


def training_state_to_bytes(training_state: Any) -> bytes:
    from flax import serialization

    return serialization.to_bytes(training_state)


def read_training_state_bytes(path: str | Path) -> bytes | None:
    path = Path(path)
    if not path.exists():
        return None
    return path.read_bytes()


def write_training_state_bytes(path: str | Path, training_state: Any) -> None:
    """Atomically saves a host-side PPO TrainingState."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = training_state_to_bytes(training_state)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_bytes(payload)
    tmp.replace(path)


__all__ = [
    "train",
    "read_training_state_bytes",
    "training_state_to_bytes",
    "write_training_state_bytes",
]
