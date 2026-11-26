"""Attach separately fitted computers without mislabelling joint qualification."""
import hashlib
import json
from pathlib import Path
import torch
from fusion.composition import RecordComputer
from fusion.full_model import FullFusionModel

HERE = Path(__file__).resolve().parent


def main():
    torch.set_num_threads(1)
    output = HERE / "evidence/optional_full"
    output.mkdir(parents=True, exist_ok=True)
    results = []
    for base_seed, numeric_seed in ((17, 1701), (29, 1702), (41, 1703)):
        base = HERE / "evidence/runs" / f"memory_seed{base_seed}" / "final.pt"
        numeric = HERE / "evidence/composition" / f"seed{numeric_seed}" / "connected.pt"
        computer = RecordComputer.from_checkpoint(numeric).eval()
        full = FullFusionModel.from_v1_checkpoint(base, computer=computer, seed=numeric_seed).eval()
        destination = output / f"full_seed{base_seed}.pt"
        if destination.exists():
            raise FileExistsError("assembled endpoint already exists")
        metadata = {"base_checkpoint_sha256": hashlib.sha256(base.read_bytes()).hexdigest(),
                    "numeric_checkpoint_sha256": hashlib.sha256(numeric.read_bytes()).hexdigest(),
                    "base_seed": base_seed, "numeric_seed": numeric_seed,
                    "qualification": "numeric component trained separately; semantic/query/character feedback projections untrained",
                    "joint_language_computation_qualified": False}
        torch.save(full.checkpoint_payload(**metadata), destination)
        restored = FullFusionModel.from_checkpoint(destination).eval()
        if restored.model_identity() != full.model_identity():
            raise ValueError("assembled checkpoint did not round-trip")
        for name, value in full.base.state_dict().items():
            if not torch.equal(value, restored.base.state_dict()[name]):
                raise ValueError("base weights changed during assembly")
        result = {**metadata, "path": str(destination.relative_to(HERE)),
                  "sha256": hashlib.sha256(destination.read_bytes()).hexdigest(),
                  "model_identity": full.model_identity(), "parameters": full.parameter_summary(),
                  "state_tensor_bytes": full.state_tensor_bytes()}
        results.append(result)
    (output / "ASSEMBLY.json").write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
