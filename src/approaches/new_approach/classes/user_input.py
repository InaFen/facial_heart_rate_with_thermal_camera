"""
Configuration loader for heart rate detection pipeline.

Reads configuration from YAML file and provides easy access to all settings.
"""

from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
import json
import yaml


def load_config_from_yaml(config_path: str = None) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def load_config_from_json(config_path: str = None) -> Dict[str, Any]:
    """Load configuration from JSON file."""
    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, 'r', encoding='utf-8') as f:
        return json.load(f)


@dataclass
class Config:
    """
    Configuration manager for the heart rate detection pipeline.
    
    Reads configuration from YAML file and provides direct access to all settings as instance variables.
    
    Example:
        config = Config.from_yaml()
        print(config.fps)
        print(config.path_video)
        print(config.forehead)
    """
    
    # Paths
    path_gray_frames: str
    path_video: str
    path_video_tracked_face: str
    path_video_thresholded_pixel: str
    path_video_facemesh: str
    path_pixels_log: str
    path_video_thermal_with_roi_masks: str
    path_roi_pixel_data: str
    path_ground_truth: str
    
    # Processing parameters
    overwrite_existing_files: bool
    plot_steps: bool
    fps: float
    bandpass_lowcut: float
    bandpass_highcut: float
    bandpass_order: int
    use_ema: bool 
    ema_alpha: float 
    manual_roi_enabled: bool = False
    manual_roi_names: List[str] = field(default_factory=lambda: ["manual_roi"])
    manual_roi_modes: List[str] = field(default_factory=lambda: ["rectangle"])
    manual_roi_mask_load_paths: Optional[List[str]] = field(default_factory=lambda: [None])
    manual_roi_mask_save_paths: Optional[List[str]] = field(default_factory=lambda: [None])
    manual_roi_selection_colormap: bool = True
    manual_roi_split_enabled: bool = False
    manual_roi_subroi_grid: List[int] = field(default_factory=lambda: [2, 2])
    
    # Timing
    time_first_frame_signal: Optional[str] = None
    time_first_frame_ecg: Optional[str] = None
    timespan_signal_to_look_at: Optional[List[float]] = None
    
    # ROI Landmarks
    whole_face: List[int] = field(default_factory=list)
    roi_landmarks_subgrid_face: Dict[str, List[int]] = field(default_factory=dict)
    
    # Config path (not serialized)
    config_path: Path = field(default=None, repr=False)

    @classmethod
    def _from_data(cls, data: Dict[str, Any], config_path: Path) -> "Config":
        """Build Config from parsed dictionary data."""
        # Extract nested data
        paths_data = data['paths']
        processing_data = data['processing']
        timing_data = data['timing']
        roi_data = data['roi_landmarks']

        return cls(
            # Paths
            path_gray_frames=paths_data['gray_frames'],
            path_video=paths_data['video'],
            path_video_tracked_face=paths_data['video_tracked_face'],
            path_video_thresholded_pixel=paths_data['video_thresholded_pixel'],
            path_video_facemesh=paths_data['video_facemesh'],
            path_pixels_log=paths_data['pixels_log'],
            path_video_thermal_with_roi_masks=paths_data['video_thermal_with_roi_masks'],
            path_roi_pixel_data=paths_data['roi_pixel_data'],
            path_ground_truth=paths_data['ground_truth'],

            # Processing
            overwrite_existing_files=processing_data['overwrite_existing_files'],
            plot_steps=processing_data['plot_steps'],
            fps=processing_data['fps'],
            bandpass_lowcut=processing_data['bandpass_lowcut'],
            bandpass_highcut=processing_data['bandpass_highcut'],
            bandpass_order=processing_data['bandpass_order'],
            use_ema=processing_data['use_ema'],
            ema_alpha=processing_data['ema_alpha'],
            manual_roi_enabled=processing_data.get('manual_roi_enabled', False),
            manual_roi_names=processing_data.get('manual_roi_names', [processing_data.get('manual_roi_name', "manual_roi")]),
            manual_roi_modes=processing_data.get('manual_roi_modes', [processing_data.get('manual_roi_mode', "rectangle")]),
            manual_roi_mask_load_paths=processing_data.get('manual_roi_mask_load_paths', [processing_data.get('manual_roi_mask_load_path', None)]),
            manual_roi_mask_save_paths=processing_data.get('manual_roi_mask_save_paths', [processing_data.get('manual_roi_mask_save_path', None)]),
            manual_roi_selection_colormap=processing_data.get('manual_roi_selection_colormap', True),
            manual_roi_split_enabled=processing_data.get('manual_roi_split_enabled', False),
            manual_roi_subroi_grid=processing_data.get('manual_roi_subroi_grid', [2, 2]),

            # Timing
            time_first_frame_signal=timing_data['time_first_frame_signal'],
            time_first_frame_ecg=timing_data['time_first_frame_ecg'],
            timespan_signal_to_look_at=timing_data.get('timespan_signal_to_look_at', None),

            # ROI Landmarks
            whole_face=roi_data['whole_face'],
            roi_landmarks_subgrid_face=data['roi_landmarks_subgrid_face'],

            # Config path
            config_path=config_path
        )

    @classmethod
    def from_yaml(cls, config_path: str = None) -> "Config":
        """Load configuration from YAML file."""
        config_path = Path(config_path)
        data = load_config_from_yaml(config_path)
        return cls._from_data(data, config_path)

    @classmethod
    def from_json(cls, config_path: str = None) -> "Config":
        """Load configuration from JSON file."""
        config_path = Path(config_path)
        data = load_config_from_json(config_path)
        return cls._from_data(data, config_path)

