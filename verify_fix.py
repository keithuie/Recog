
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from gui.desktop.core.alarm_manager import AlarmManager, AlarmCondition, AlarmConditionType, AlarmSeverity

def test_alarm_creation():
    manager = AlarmManager()
    
    # Simulate data from wizard
    condition_data = {
        'type': 'MATCH_BELOW',
        'severity': 'CRITICAL',
        'threshold': 75.0
    }
    
    # Logic from app.py
    try:
        cond_type = AlarmConditionType[condition_data['type']]
        severity = AlarmSeverity[condition_data['severity']]
        
        new_condition = AlarmCondition(
            name=f"{condition_data['severity']} Alert",
            condition_type=cond_type,
            threshold=condition_data['threshold'],
            severity=severity
        )
        manager.add_condition(new_condition)
        print("Successfully added condition")
    except Exception as e:
        print(f"Failed to add condition: {e}")
        sys.exit(1)

if __name__ == "__main__":
    test_alarm_creation()
