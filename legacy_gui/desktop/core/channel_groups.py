"""
Channel Grouping for Multivariate Pattern Matching

Allows users to group related channels together so the ML model
can detect patterns across multiple channels simultaneously.
"""

import json
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any
from pathlib import Path


@dataclass
class ChannelGroup:
    """
    A named group of channels for multivariate analysis.

    When channels are grouped, the detector learns joint patterns
    across all channels in the group, enabling detection of
    multi-channel anomalies that might not be visible in
    individual channel analysis.
    """
    name: str
    channels: List[str] = field(default_factory=list)
    description: str = ""
    color: str = "#007AFF"
    enabled: bool = True

    def add_channel(self, channel_name: str):
        """Add a channel to this group"""
        if channel_name not in self.channels:
            self.channels.append(channel_name)

    def remove_channel(self, channel_name: str):
        """Remove a channel from this group"""
        if channel_name in self.channels:
            self.channels.remove(channel_name)

    def has_channel(self, channel_name: str) -> bool:
        """Check if channel is in this group"""
        return channel_name in self.channels

    @property
    def channel_count(self) -> int:
        """Number of channels in group"""
        return len(self.channels)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'name': self.name,
            'channels': self.channels,
            'description': self.description,
            'color': self.color,
            'enabled': self.enabled,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ChannelGroup':
        """Create from dictionary"""
        return cls(
            name=data['name'],
            channels=data.get('channels', []),
            description=data.get('description', ''),
            color=data.get('color', '#007AFF'),
            enabled=data.get('enabled', True),
        )


class ChannelGroupManager:
    """
    Manages channel groups for the detector.

    Provides functionality to:
    - Create, edit, delete channel groups
    - Save/load group configurations
    - Get channels for detector configuration
    """

    def __init__(self):
        self._groups: Dict[str, ChannelGroup] = {}
        self._ungrouped_channels: List[str] = []

    @property
    def groups(self) -> List[ChannelGroup]:
        """Get all groups"""
        return list(self._groups.values())

    @property
    def enabled_groups(self) -> List[ChannelGroup]:
        """Get only enabled groups"""
        return [g for g in self._groups.values() if g.enabled]

    @property
    def ungrouped_channels(self) -> List[str]:
        """Get channels not assigned to any group"""
        return self._ungrouped_channels.copy()

    def create_group(self, name: str, channels: List[str] = None,
                     description: str = "", color: str = "#007AFF") -> ChannelGroup:
        """Create a new channel group"""
        if name in self._groups:
            raise ValueError(f"Group '{name}' already exists")

        group = ChannelGroup(
            name=name,
            channels=channels or [],
            description=description,
            color=color,
        )
        self._groups[name] = group

        # Remove channels from ungrouped list
        for ch in group.channels:
            if ch in self._ungrouped_channels:
                self._ungrouped_channels.remove(ch)

        return group

    def get_group(self, name: str) -> Optional[ChannelGroup]:
        """Get a group by name"""
        return self._groups.get(name)

    def delete_group(self, name: str):
        """Delete a group, moving its channels to ungrouped"""
        if name in self._groups:
            group = self._groups[name]
            self._ungrouped_channels.extend(group.channels)
            del self._groups[name]

    def rename_group(self, old_name: str, new_name: str):
        """Rename a group"""
        if old_name not in self._groups:
            raise ValueError(f"Group '{old_name}' not found")
        if new_name in self._groups:
            raise ValueError(f"Group '{new_name}' already exists")

        group = self._groups[old_name]
        group.name = new_name
        self._groups[new_name] = group
        del self._groups[old_name]

    def add_channel_to_group(self, channel_name: str, group_name: str):
        """Add a channel to a specific group"""
        if group_name not in self._groups:
            raise ValueError(f"Group '{group_name}' not found")

        # Remove from other groups first
        for group in self._groups.values():
            group.remove_channel(channel_name)

        # Remove from ungrouped
        if channel_name in self._ungrouped_channels:
            self._ungrouped_channels.remove(channel_name)

        # Add to target group
        self._groups[group_name].add_channel(channel_name)

    def remove_channel_from_group(self, channel_name: str, group_name: str):
        """Remove a channel from a group (moves to ungrouped)"""
        if group_name in self._groups:
            self._groups[group_name].remove_channel(channel_name)
            if channel_name not in self._ungrouped_channels:
                self._ungrouped_channels.append(channel_name)

    def set_available_channels(self, channels: List[str]):
        """Set the list of available channels from data source"""
        # Find all channels already in groups
        grouped_channels = set()
        for group in self._groups.values():
            grouped_channels.update(group.channels)

        # Add new channels to ungrouped
        self._ungrouped_channels = [
            ch for ch in channels if ch not in grouped_channels
        ]

        # Remove non-existent channels from groups
        valid_channels = set(channels)
        for group in self._groups.values():
            group.channels = [ch for ch in group.channels if ch in valid_channels]

    def get_channel_group(self, channel_name: str) -> Optional[ChannelGroup]:
        """Get the group containing a specific channel"""
        for group in self._groups.values():
            if channel_name in group.channels:
                return group
        return None

    def get_detector_channel_configs(self, channel_ranges: Dict[str, tuple]) -> List[Dict]:
        """
        Get channel configurations for the MIQ detector.

        Args:
            channel_ranges: Dict mapping channel names to (min, max) tuples

        Returns:
            List of channel config dicts for detector.configure_channels()
        """
        configs = []

        for group in self.enabled_groups:
            for channel in group.channels:
                if channel in channel_ranges:
                    min_val, max_val = channel_ranges[channel]
                    configs.append({
                        'name': channel,
                        'min_value': min_val,
                        'max_value': max_val,
                        'group': group.name,
                    })

        # Add ungrouped channels
        for channel in self._ungrouped_channels:
            if channel in channel_ranges:
                min_val, max_val = channel_ranges[channel]
                configs.append({
                    'name': channel,
                    'min_value': min_val,
                    'max_value': max_val,
                    'group': None,
                })

        return configs

    def save_to_file(self, filepath: str):
        """Save group configuration to JSON file"""
        data = {
            'groups': [g.to_dict() for g in self._groups.values()],
            'ungrouped_channels': self._ungrouped_channels,
        }

        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)

    def load_from_file(self, filepath: str):
        """Load group configuration from JSON file"""
        with open(filepath, 'r') as f:
            data = json.load(f)

        self._groups = {}
        for group_data in data.get('groups', []):
            group = ChannelGroup.from_dict(group_data)
            self._groups[group.name] = group

        self._ungrouped_channels = data.get('ungrouped_channels', [])

    def clear(self):
        """Clear all groups"""
        self._groups.clear()
        self._ungrouped_channels.clear()
