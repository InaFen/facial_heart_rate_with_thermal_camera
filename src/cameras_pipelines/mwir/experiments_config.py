from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import yaml


@dataclass(frozen=True)
class ExperimentSelection:
	"""Typed experiment configuration selected from MWIR experiments YAML."""
	asc_directory: Path
	cache_path: Path
	ground_truth_path: Path
	ground_truth_txt_path: Path
	offset_ground_truth_seconds: float
	video_output_path: Path


def _coerce_yaml_path(value: str | Path) -> Path:
	"""Convert YAML string/path values to Path, fixing common missing-leading-slash cases."""
	path = Path(value)
	if not path.is_absolute() and str(value).startswith("mnt/"):
		return Path("/") / str(value)
	return path


def load_experiment_from_yaml(
	yaml_path: str | Path,
	person: str,
	region: Literal["temple", "face"],
	phase: Literal["before", "after"],
) -> ExperimentSelection:
	"""Load one MWIR experiment selection from YAML."""
	yaml_path = Path(yaml_path)
	if not yaml_path.exists():
		raise FileNotFoundError(f"Experiments YAML does not exist: {yaml_path}")

	with yaml_path.open("r", encoding="utf-8") as file_handle:
		data = yaml.safe_load(file_handle)

	if not isinstance(data, dict):
		raise ValueError(f"Expected top-level mapping in YAML: {yaml_path}")

	person_key = person.strip().lower()
	if person_key not in data:
		available_people = ", ".join(sorted(data.keys()))
		raise ValueError(f"Unknown person '{person}'. Available: {available_people}")

	person_data = data[person_key]
	if not isinstance(person_data, dict):
		raise ValueError(f"Expected mapping for person '{person_key}' in {yaml_path}")

	suffix = f"{region}_{phase}"
	required_keys = {
		"asc_directory": f"asc_directory_{suffix}",
		"cache_path": f"cache_path_{suffix}",
		"ground_truth_path": f"ground_truth_path_{suffix}",
		"ground_truth_txt_path": f"ground_truth_txt_path_{suffix}",
		"offset_ground_truth_seconds": f"offset_ground_truth_seconds_{suffix}",
		"video_output_path": f"video_output_path_{suffix}",
	}

	missing_keys = [yaml_key for yaml_key in required_keys.values() if yaml_key not in person_data]
	if missing_keys:
		raise ValueError(
			f"Missing keys for selection person='{person_key}', region='{region}', phase='{phase}': {missing_keys}"
		)

	asc_directory = _coerce_yaml_path(person_data[required_keys["asc_directory"]])
	cache_path = _coerce_yaml_path(person_data[required_keys["cache_path"]])
	ground_truth_path = _coerce_yaml_path(person_data[required_keys["ground_truth_path"]])
	ground_truth_txt_path = _coerce_yaml_path(person_data[required_keys["ground_truth_txt_path"]])
	video_output_path = _coerce_yaml_path(person_data[required_keys["video_output_path"]])
	offset_ground_truth_seconds = float(person_data[required_keys["offset_ground_truth_seconds"]])

	return ExperimentSelection(
		asc_directory=asc_directory,
		cache_path=cache_path,
		ground_truth_path=ground_truth_path,
		ground_truth_txt_path=ground_truth_txt_path,
		offset_ground_truth_seconds=offset_ground_truth_seconds,
		video_output_path=video_output_path,
	)
