"""
Alarm Management System

Handles:
- Alarm condition definitions
- Threshold monitoring
- Email notifications
- Alarm history and acknowledgment
"""

import json
import smtplib
import threading
from dataclasses import dataclass, field
from datetime import datetime
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from enum import Enum, auto
from pathlib import Path
from typing import List, Dict, Optional, Any, Callable
from queue import Queue


class AlarmSeverity(Enum):
    """Alarm severity levels"""
    INFO = auto()
    WARNING = auto()
    CRITICAL = auto()


class AlarmConditionType(Enum):
    """Types of alarm conditions"""
    MATCH_BELOW = auto()  # Match score below threshold
    MATCH_ABOVE = auto()  # Match score above threshold (unusual)
    CHANNEL_ABOVE = auto()  # Channel value above threshold
    CHANNEL_BELOW = auto()  # Channel value below threshold
    RATE_OF_CHANGE = auto()  # Rapid change detection


@dataclass
class AlarmCondition:
    """Definition of an alarm trigger condition"""
    name: str
    condition_type: AlarmConditionType
    threshold: float
    channel: Optional[str] = None  # For channel-specific conditions
    severity: AlarmSeverity = AlarmSeverity.WARNING
    enabled: bool = True
    description: str = ""

    def evaluate(self, match_score: float = None,
                 channel_values: Dict[str, float] = None) -> bool:
        """
        Evaluate if this condition is triggered.

        Args:
            match_score: Current match score (0-100)
            channel_values: Dict of channel name -> current value

        Returns:
            True if alarm should trigger
        """
        if not self.enabled:
            return False

        if self.condition_type == AlarmConditionType.MATCH_BELOW:
            return match_score is not None and match_score < self.threshold

        elif self.condition_type == AlarmConditionType.MATCH_ABOVE:
            return match_score is not None and match_score > self.threshold

        elif self.condition_type == AlarmConditionType.CHANNEL_ABOVE:
            if self.channel and channel_values:
                value = channel_values.get(self.channel)
                return value is not None and value > self.threshold

        elif self.condition_type == AlarmConditionType.CHANNEL_BELOW:
            if self.channel and channel_values:
                value = channel_values.get(self.channel)
                return value is not None and value < self.threshold

        return False

    def to_dict(self) -> Dict[str, Any]:
        return {
            'name': self.name,
            'condition_type': self.condition_type.name,
            'threshold': self.threshold,
            'channel': self.channel,
            'severity': self.severity.name,
            'enabled': self.enabled,
            'description': self.description,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AlarmCondition':
        return cls(
            name=data['name'],
            condition_type=AlarmConditionType[data['condition_type']],
            threshold=data['threshold'],
            channel=data.get('channel'),
            severity=AlarmSeverity[data.get('severity', 'WARNING')],
            enabled=data.get('enabled', True),
            description=data.get('description', ''),
        )


@dataclass
class Alarm:
    """An active or historical alarm instance"""
    id: str
    condition_name: str
    severity: AlarmSeverity
    timestamp: datetime
    message: str
    match_score: Optional[float] = None
    channel_values: Dict[str, float] = field(default_factory=dict)
    acknowledged: bool = False
    acknowledged_by: Optional[str] = None
    acknowledged_at: Optional[datetime] = None
    cleared: bool = False
    cleared_at: Optional[datetime] = None

    def acknowledge(self, user: str = "System"):
        """Acknowledge the alarm"""
        self.acknowledged = True
        self.acknowledged_by = user
        self.acknowledged_at = datetime.now()

    def clear(self):
        """Clear the alarm"""
        self.cleared = True
        self.cleared_at = datetime.now()

    def to_dict(self) -> Dict[str, Any]:
        return {
            'id': self.id,
            'condition_name': self.condition_name,
            'severity': self.severity.name,
            'timestamp': self.timestamp.isoformat(),
            'message': self.message,
            'match_score': self.match_score,
            'channel_values': self.channel_values,
            'acknowledged': self.acknowledged,
            'acknowledged_by': self.acknowledged_by,
            'acknowledged_at': self.acknowledged_at.isoformat() if self.acknowledged_at else None,
            'cleared': self.cleared,
            'cleared_at': self.cleared_at.isoformat() if self.cleared_at else None,
        }


@dataclass
class EmailConfig:
    """Email notification configuration"""
    smtp_server: str = "smtp.gmail.com"
    smtp_port: int = 587
    use_tls: bool = True
    username: str = ""
    password: str = ""
    from_address: str = ""
    enabled: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            'smtp_server': self.smtp_server,
            'smtp_port': self.smtp_port,
            'use_tls': self.use_tls,
            'username': self.username,
            'from_address': self.from_address,
            'enabled': self.enabled,
            # Don't save password in plain text
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'EmailConfig':
        return cls(
            smtp_server=data.get('smtp_server', 'smtp.gmail.com'),
            smtp_port=data.get('smtp_port', 587),
            use_tls=data.get('use_tls', True),
            username=data.get('username', ''),
            from_address=data.get('from_address', ''),
            enabled=data.get('enabled', False),
        )


class AlarmManager:
    """
    Manages alarm conditions, notifications, and history.
    """

    def __init__(self):
        self._conditions: Dict[str, AlarmCondition] = {}
        self._active_alarms: Dict[str, Alarm] = {}
        self._alarm_history: List[Alarm] = []
        self._notification_emails: List[str] = []
        self._email_config = EmailConfig()
        self._email_queue: Queue = Queue()
        self._email_thread: Optional[threading.Thread] = None
        self._alarm_counter: int = 0
        self._on_alarm_callbacks: List[Callable[[Alarm], None]] = []
        self._suppression_interval: float = 60.0  # Seconds between repeat alarms
        self._last_alarm_times: Dict[str, datetime] = {}

    @property
    def conditions(self) -> List[AlarmCondition]:
        return list(self._conditions.values())

    @property
    def active_alarms(self) -> List[Alarm]:
        return [a for a in self._active_alarms.values() if not a.cleared]

    @property
    def alarm_history(self) -> List[Alarm]:
        return self._alarm_history.copy()

    @property
    def notification_emails(self) -> List[str]:
        return self._notification_emails.copy()

    def add_condition(self, condition: AlarmCondition):
        """Add an alarm condition"""
        self._conditions[condition.name] = condition

    def remove_condition(self, name: str):
        """Remove an alarm condition"""
        if name in self._conditions:
            del self._conditions[name]

    def get_condition(self, name: str) -> Optional[AlarmCondition]:
        return self._conditions.get(name)

    def update_condition(self, name: str, **kwargs):
        """Update condition properties"""
        if name in self._conditions:
            condition = self._conditions[name]
            for key, value in kwargs.items():
                if hasattr(condition, key):
                    setattr(condition, key, value)

    def add_notification_email(self, email: str):
        """Add email address for notifications"""
        if email and email not in self._notification_emails:
            self._notification_emails.append(email)

    def remove_notification_email(self, email: str):
        """Remove email from notifications"""
        if email in self._notification_emails:
            self._notification_emails.remove(email)

    def set_email_config(self, config: EmailConfig):
        """Set email server configuration"""
        self._email_config = config

    def add_alarm_callback(self, callback: Callable[[Alarm], None]):
        """Add callback for when alarms trigger"""
        self._on_alarm_callbacks.append(callback)

    def evaluate(self, match_score: float, channel_values: Dict[str, float] = None):
        """
        Evaluate all conditions and trigger alarms as needed.

        Args:
            match_score: Current match score (0-100)
            channel_values: Current channel values
        """
        now = datetime.now()

        for condition in self._conditions.values():
            if condition.evaluate(match_score, channel_values):
                # Check suppression
                last_time = self._last_alarm_times.get(condition.name)
                if last_time:
                    elapsed = (now - last_time).total_seconds()
                    if elapsed < self._suppression_interval:
                        continue

                # Create alarm
                self._trigger_alarm(condition, match_score, channel_values, now)

    def _trigger_alarm(self, condition: AlarmCondition, match_score: float,
                       channel_values: Dict[str, float], timestamp: datetime):
        """Trigger an alarm"""
        self._alarm_counter += 1
        alarm_id = f"ALM-{self._alarm_counter:06d}"

        # Build message
        if condition.condition_type == AlarmConditionType.MATCH_BELOW:
            message = f"Match score {match_score:.1f}% below threshold {condition.threshold}%"
        elif condition.condition_type == AlarmConditionType.CHANNEL_ABOVE:
            value = channel_values.get(condition.channel, 0) if channel_values else 0
            message = f"Channel '{condition.channel}' value {value:.2f} above threshold {condition.threshold}"
        elif condition.condition_type == AlarmConditionType.CHANNEL_BELOW:
            value = channel_values.get(condition.channel, 0) if channel_values else 0
            message = f"Channel '{condition.channel}' value {value:.2f} below threshold {condition.threshold}"
        else:
            message = f"Alarm condition '{condition.name}' triggered"

        alarm = Alarm(
            id=alarm_id,
            condition_name=condition.name,
            severity=condition.severity,
            timestamp=timestamp,
            message=message,
            match_score=match_score,
            channel_values=channel_values.copy() if channel_values else {},
        )

        self._active_alarms[alarm_id] = alarm
        self._alarm_history.append(alarm)
        self._last_alarm_times[condition.name] = timestamp

        # Notify callbacks
        for callback in self._on_alarm_callbacks:
            try:
                callback(alarm)
            except Exception as e:
                print(f"Alarm callback error: {e}")

        # Queue email notification
        if self._email_config.enabled and self._notification_emails:
            self._email_queue.put(alarm)
            self._ensure_email_thread()

    def acknowledge_alarm(self, alarm_id: str, user: str = "System"):
        """Acknowledge an alarm"""
        if alarm_id in self._active_alarms:
            self._active_alarms[alarm_id].acknowledge(user)

    def clear_alarm(self, alarm_id: str):
        """Clear an alarm"""
        if alarm_id in self._active_alarms:
            self._active_alarms[alarm_id].clear()

    def clear_all_alarms(self):
        """Clear all active alarms"""
        for alarm in self._active_alarms.values():
            alarm.clear()

    def _ensure_email_thread(self):
        """Ensure email sending thread is running"""
        if self._email_thread is None or not self._email_thread.is_alive():
            self._email_thread = threading.Thread(target=self._email_loop, daemon=True)
            self._email_thread.start()

    def _email_loop(self):
        """Background thread for sending emails"""
        while True:
            try:
                alarm = self._email_queue.get(timeout=5.0)
                self._send_alarm_email(alarm)
            except Exception:
                break  # Queue empty timeout

    def _send_alarm_email(self, alarm: Alarm):
        """Send email notification for an alarm"""
        if not self._email_config.enabled or not self._notification_emails:
            return

        try:
            # Build email
            subject = f"[MachineIQ {alarm.severity.name}] {alarm.condition_name}"

            body = f"""
MachineIQ Alarm Notification

Alarm ID: {alarm.id}
Condition: {alarm.condition_name}
Severity: {alarm.severity.name}
Time: {alarm.timestamp.strftime('%Y-%m-%d %H:%M:%S')}

Message: {alarm.message}

Match Score: {alarm.match_score:.1f}%

This is an automated notification from MachineIQ condition monitoring system.
            """

            msg = MIMEMultipart()
            msg['Subject'] = subject
            msg['From'] = self._email_config.from_address
            msg['To'] = ', '.join(self._notification_emails)
            msg.attach(MIMEText(body, 'plain'))

            # Send
            with smtplib.SMTP(self._email_config.smtp_server, self._email_config.smtp_port) as server:
                if self._email_config.use_tls:
                    server.starttls()
                if self._email_config.username and self._email_config.password:
                    server.login(self._email_config.username, self._email_config.password)
                server.sendmail(
                    self._email_config.from_address,
                    self._notification_emails,
                    msg.as_string()
                )

        except Exception as e:
            print(f"Email send error: {e}")

    def save_config(self, filepath: str):
        """Save alarm configuration to file"""
        data = {
            'conditions': [c.to_dict() for c in self._conditions.values()],
            'notification_emails': self._notification_emails,
            'email_config': self._email_config.to_dict(),
            'suppression_interval': self._suppression_interval,
        }

        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)

    def load_config(self, filepath: str):
        """Load alarm configuration from file"""
        with open(filepath, 'r') as f:
            data = json.load(f)

        self._conditions = {}
        for cond_data in data.get('conditions', []):
            condition = AlarmCondition.from_dict(cond_data)
            self._conditions[condition.name] = condition

        self._notification_emails = data.get('notification_emails', [])
        self._email_config = EmailConfig.from_dict(data.get('email_config', {}))
        self._suppression_interval = data.get('suppression_interval', 60.0)

    def create_default_conditions(self):
        """Create default alarm conditions"""
        # Critical low match score
        self.add_condition(AlarmCondition(
            name="Critical Match Score",
            condition_type=AlarmConditionType.MATCH_BELOW,
            threshold=70.0,
            severity=AlarmSeverity.CRITICAL,
            description="Match score dropped below critical threshold",
        ))

        # Warning low match score
        self.add_condition(AlarmCondition(
            name="Low Match Score",
            condition_type=AlarmConditionType.MATCH_BELOW,
            threshold=85.0,
            severity=AlarmSeverity.WARNING,
            description="Match score below warning threshold",
        ))
