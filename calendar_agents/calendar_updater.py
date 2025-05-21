#!/usr/bin/env python3
# calendar_updater.py - Utilities for calendar operations

import json
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional

class CalendarUpdater:
    """Utility class to handle calendar operations."""
    
    def __init__(self, schedule_file: Path, output_file: Path):
        self.schedule_file = schedule_file
        self.output_file = output_file
        self.calendar_data = self._load_schedule()
    
    def _load_schedule(self) -> Dict[str, Any]:
        """Load the current schedule from file."""
        with open(self.schedule_file, 'r') as f:
            return json.load(f)
    
    def save_calendar(self) -> None:
        """Save the updated calendar to output file."""
        with open(self.output_file, 'w') as f:
            json.dump(self.calendar_data, f, indent=2)
        
    def get_events(self) -> List[Dict[str, Any]]:
        """Get all events from the calendar."""
        return self.calendar_data.get("schedule", [])
    
    def get_user_preferences(self) -> Dict[str, Any]:
        """Get user preferences from the calendar."""
        return self.calendar_data.get("user_preferences", {})
    
    def add_event(self, title: str, description: str, start: str, end: str, 
                  location: str = "", priority: str = "medium") -> Dict[str, Any]:
        """Add a new event to the calendar."""
        # Generate a unique ID for the event
        event_id = str(uuid.uuid4())[:8]
        
        # Create the event object
        event = {
            "id": event_id,
            "title": title,
            "description": description,
            "start": start,
            "end": end,
            "location": location,
            "priority": priority
        }
        
        # Add to the schedule
        self.calendar_data["schedule"].append(event)
        
        return event
    
    def update_event(self, event_id: str, **kwargs) -> Optional[Dict[str, Any]]:
        """Update an existing event by ID."""
        for i, event in enumerate(self.calendar_data["schedule"]):
            if event["id"] == event_id:
                # Update the fields provided in kwargs
                for key, value in kwargs.items():
                    if key in event:
                        event[key] = value
                
                # Update the event in the calendar
                self.calendar_data["schedule"][i] = event
                return event
        
        return None  # Event not found
    
    def delete_event(self, event_id: str) -> bool:
        """Delete an event by ID."""
        for i, event in enumerate(self.calendar_data["schedule"]):
            if event["id"] == event_id:
                del self.calendar_data["schedule"][i]
                return True
        
        return False  # Event not found
    
    def find_available_slots(self, duration_minutes: int = 60, 
                            start_date: Optional[str] = None, 
                            end_date: Optional[str] = None,
                            respect_preferences: bool = True) -> List[Dict[str, str]]:
        """Find available time slots in the calendar."""
        # Logic to find available slots based on existing events and duration
        # This would involve checking for gaps between events
        # Respecting user preferences like work hours and focus time
        
        # For demonstration, return a dummy available slot
        return [
            {
                "start": "2023-10-17T10:00:00",
                "end": "2023-10-17T11:00:00"
            },
            {
                "start": "2023-10-17T15:00:00",
                "end": "2023-10-17T16:00:00"
            },
            {
                "start": "2023-10-19T09:30:00",
                "end": "2023-10-19T10:30:00"
            }
        ]
    
    def check_conflicts(self, start: str, end: str) -> List[Dict[str, Any]]:
        """Check for conflicts with existing events."""
        conflicts = []
        
        start_dt = datetime.fromisoformat(start)
        end_dt = datetime.fromisoformat(end)
        
        for event in self.calendar_data["schedule"]:
            event_start = datetime.fromisoformat(event["start"])
            event_end = datetime.fromisoformat(event["end"])
            
            # Check if there's an overlap
            if (start_dt < event_end and end_dt > event_start):
                conflicts.append(event)
        
        return conflicts
    
    def export_to_ical(self, output_file: Optional[Path] = None) -> str:
        """Export the calendar to iCalendar format."""
        # In a real implementation, this would convert to iCal format
        # For this demo, we'll just return a placeholder message
        output = output_file or self.output_file.with_suffix('.ics')
        return f"Calendar would be exported to {output}"
    
    def get_calendar_summary(self) -> Dict[str, Any]:
        """Get a summary of the calendar."""
        events = self.get_events()
        
        summary = {
            "total_events": len(events),
            "events_by_priority": {
                "critical": 0,
                "high": 0,
                "medium": 0,
                "low": 0
            },
            "upcoming_events": []
        }
        
        # Count events by priority
        for event in events:
            priority = event.get("priority", "medium")
            if priority in summary["events_by_priority"]:
                summary["events_by_priority"][priority] += 1
        
        # Get upcoming events (next 7 days)
        now = datetime.now()
        one_week_later = now + timedelta(days=7)
        
        for event in events:
            event_time = datetime.fromisoformat(event["start"])
            if now <= event_time <= one_week_later:
                summary["upcoming_events"].append(event)
        
        return summary 