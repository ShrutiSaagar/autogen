"""
This example demonstrates an approach one can use to
implement a async human in the loop system.
The system consists of two agents:
1. An assistant agent that uses a tool call to schedule a meeting (this is a mock)
2. A user proxy that is used as a proxy for a slow human user. When this user receives
a message from the assistant, it sends out a termination request with the query for the real human.
The query to the human is sent out (as an input to the terminal here, but it could be an email or
anything else) and the state of the runtime is saved in a persistent layer. When the user responds,
the runtime is rehydrated with the state and the user input is sent back to the runtime.

This is a simple example that can be extended to more complex scenarios as well.
Whenever implementing a human in the loop system, it is important to consider that human looped
systems can be slow - Humans take time to respond, but also depending on your medium of
communication, the time taken can vary significantly. When waiting for the human to respond, it is
possible that the system may be torn down. In such cases, it is important to save the state of the
system with any relevant information that is needed to rehydrate the system. When designing such
systems, it can be helpful recognize the trade-offs at which point to save the system state.
In the given (simple) example, the system state is saved when the user input is needed. However, in
a more complex system, it may be necessary to save the state at multiple points to ensure that the
system can be rehydrated to the correct state.
Additionally, we use "human"-in-loop in this example, but the same principles can be applied to any
slow external system that the agent needs to interact with.
"""

import asyncio
import datetime
import json
import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional

from autogen_core import (
    CancellationToken,
    DefaultInterventionHandler,
    DefaultTopicId,
    FunctionCall,
    MessageContext,
    RoutedAgent,
    SingleThreadedAgentRuntime,
    message_handler,
    type_subscription,
)
from autogen_core.model_context import BufferedChatCompletionContext
from autogen_core.models import (
    AssistantMessage,
    ChatCompletionClient,
    SystemMessage,
    UserMessage,
)
from autogen_core.tools import BaseTool
from pydantic import BaseModel, Field
import yaml

# Google Calendar API imports
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from googleapiclient.errors import HttpError
import pickle


@dataclass
class TextMessage:
    source: str
    content: str


@dataclass
class UserTextMessage(TextMessage):
    pass


@dataclass
class AssistantTextMessage(TextMessage):
    pass


@dataclass
class GetSlowUserMessage:
    content: str


@dataclass
class TerminateMessage:
    content: str


class MockPersistence:
    def __init__(self):
        self._content: Mapping[str, Any] = {}

    def load_content(self) -> Mapping[str, Any]:
        return self._content

    def save_content(self, content: Mapping[str, Any]) -> None:
        self._content = content


state_persister = MockPersistence()


@type_subscription("scheduling_assistant_conversation")
class SlowUserProxyAgent(RoutedAgent):
    def __init__(
        self,
        name: str,
        description: str,
    ) -> None:
        super().__init__(description)
        self._model_context = BufferedChatCompletionContext(buffer_size=5)
        self._name = name

    @message_handler
    async def handle_message(self, message: AssistantTextMessage, ctx: MessageContext) -> None:
        await self._model_context.add_message(AssistantMessage(content=message.content, source=message.source))
        await self.publish_message(
            GetSlowUserMessage(content=message.content), topic_id=DefaultTopicId("scheduling_assistant_conversation")
        )

    async def save_state(self) -> Mapping[str, Any]:
        state_to_save = {
            "memory": await self._model_context.save_state(),
        }
        return state_to_save

    async def load_state(self, state: Mapping[str, Any]) -> None:
        await self._model_context.load_state(state["memory"])


class ScheduleMeetingInput(BaseModel):
    recipient: str = Field(description="Name of recipient")
    date: str = Field(description="Date of meeting")
    time: str = Field(description="Time of meeting")
    duration_minutes: int = Field(description="Duration of meeting in minutes", default=30)
    summary: str = Field(description="Meeting summary/title", default="Scheduled Meeting")
    description: str = Field(description="Meeting description", default="")
    location: str = Field(description="Meeting location", default="")
    attendee_email: str = Field(description="Email of the attendee", default="")


class ScheduleMeetingOutput(BaseModel):
    event_id: str = Field(description="ID of the created calendar event")
    event_link: str = Field(description="Link to the created calendar event")


class MultipleEventsInput(BaseModel):
    events: list[ScheduleMeetingInput] = Field(description="List of events to create")
    batch_description: str = Field(description="Description of the batch operation", default="Multiple events creation")


class MultipleEventsOutput(BaseModel):
    created_events: list[ScheduleMeetingOutput] = Field(description="List of successfully created events")
    failed_events: list[dict] = Field(description="List of events that failed to create with error details")
    total_created: int = Field(description="Total number of successfully created events")
    total_failed: int = Field(description="Total number of failed events")
    batch_summary: str = Field(description="Summary of the batch operation")


class GetEventsInput(BaseModel):
    start_date: str = Field(description="Start date for event search (YYYY-MM-DD format)")
    end_date: str = Field(description="End date for event search (YYYY-MM-DD format)")
    max_results: int = Field(description="Maximum number of events to return", default=20)


class CalendarEvent(BaseModel):
    event_id: str = Field(description="ID of the calendar event")
    summary: str = Field(description="Event title/summary")
    start_time: str = Field(description="Event start time")
    end_time: str = Field(description="Event end time")
    location: str = Field(description="Event location", default="")
    description: str = Field(description="Event description", default="")
    attendees: list = Field(description="List of attendees", default=[])


class GetEventsOutput(BaseModel):
    events: list[CalendarEvent] = Field(description="List of calendar events")
    total_count: int = Field(description="Total number of events found")


class EditEventInput(BaseModel):
    event_id: str = Field(description="ID of the event to edit")
    summary: str = Field(description="New event title/summary", default="")
    start_date: str = Field(description="New start date (YYYY-MM-DD)", default="")
    start_time: str = Field(description="New start time (HH:MM)", default="")
    duration_minutes: int = Field(description="New duration in minutes", default=0)
    location: str = Field(description="New event location", default="")
    description: str = Field(description="New event description", default="")


class EditEventOutput(BaseModel):
    success: bool = Field(description="Whether the edit was successful")
    message: str = Field(description="Success or error message")
    event_link: str = Field(description="Link to the updated event", default="")


class CreatePlaceholderEventInput(BaseModel):
    title: str = Field(description="Meeting title", default="Placeholder Meeting")
    date: str = Field(description="Date of meeting (YYYY-MM-DD format)")
    time: str = Field(description="Time of meeting (HH:MM format)")
    duration_minutes: int = Field(description="Duration in minutes", default=30)
    recipient_placeholder: str = Field(description="Placeholder for recipient", default="TBD - Recipient to be confirmed")
    location: str = Field(description="Meeting location", default="TBD - Location to be confirmed")
    description: str = Field(description="Meeting description", default="")
    is_placeholder: bool = Field(description="Mark as placeholder event", default=True)


class CreatePlaceholderEventOutput(BaseModel):
    event_id: str = Field(description="ID of the created placeholder event")
    event_link: str = Field(description="Link to the created event")
    placeholder_details: dict = Field(description="Details of what needs confirmation")


class GoogleCalendarService:
    # If modifying these scopes, delete the file token.pickle.
    SCOPES = ['https://www.googleapis.com/auth/calendar']

    def __init__(self):
        self.service = self._get_calendar_service()

    def _get_calendar_service(self):
        """Get an authorized Google Calendar API service instance."""
        creds = None
        # The file token.pickle stores the user's access and refresh tokens
        if os.path.exists('token.pickle'):
            with open('token.pickle', 'rb') as token:
                creds = pickle.load(token)
                
        # If there are no (valid) credentials available, let the user log in.
        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                creds.refresh(Request())
            else:
                flow = InstalledAppFlow.from_client_secrets_file(
                    'credentials.json', self.SCOPES)
                creds = flow.run_local_server(port=0)
            # Save the credentials for the next run
            with open('token.pickle', 'wb') as token:
                pickle.dump(creds, token)

        return build('calendar', 'v3', credentials=creds)

    def create_event(self, args: ScheduleMeetingInput) -> Dict[str, str]:
        """Create a new calendar event."""
        try:
            # Parse date and time into datetime
            datetime_str = f"{args.date} {args.time}"
            start_time = datetime.datetime.strptime(datetime_str, "%Y-%m-%d %H:%M")
            end_time = start_time + datetime.timedelta(minutes=args.duration_minutes)
            
            # Format datetime for Google Calendar API
            start_time_str = start_time.isoformat()
            end_time_str = end_time.isoformat()
            
            # Create event body
            event_body = {
                'summary': args.summary,
                'location': args.location,
                'description': args.description,
                'start': {
                    'dateTime': start_time_str,
                    'timeZone': 'America/Chicago',  # You may want to make this configurable
                },
                'end': {
                    'dateTime': end_time_str,
                    'timeZone': 'America/Chicago',  # You may want to make this configurable
                },
            }
            
            # Add attendees if email is provided
            if args.attendee_email:
                event_body['attendees'] = [{'email': args.attendee_email}]
                
            # Create the event
            event = self.service.events().insert(calendarId='primary', body=event_body).execute()
            
            return {
                'event_id': event.get('id'),
                'event_link': event.get('htmlLink')
            }
            
        except HttpError as error:
            print(f"An error occurred: {error}")
            raise

    def get_events_approx(self, args: GetEventsInput) -> Dict[str, Any]:
        """Get calendar events for a given timeframe."""
        try:
            # Parse start and end dates
            start_date = datetime.datetime.strptime(args.start_date, "%Y-%m-%d")
            end_date = datetime.datetime.strptime(args.end_date, "%Y-%m-%d")
            # set start date to 7 days before today
            start_date = start_date - datetime.timedelta(days=7)
            # set end date to 7 days after today
            end_date = end_date + datetime.timedelta(days=7)
            args.start_date = start_date.strftime("%Y-%m-%d")
            args.end_date = end_date.strftime("%Y-%m-%d")
            return self.get_events(args)
        except HttpError as error:
            print(f"An error occurred: {error}")
            raise
        except ValueError as error:
            print(f"Date parsing error: {error}")
            raise

    def get_events(self, args: GetEventsInput) -> Dict[str, Any]:
        """Get calendar events for a given timeframe."""
        try:
            # Parse start and end dates
            start_date = datetime.datetime.strptime(args.start_date, "%Y-%m-%d")
            end_date = datetime.datetime.strptime(args.end_date, "%Y-%m-%d")
            
            # Add time to make it end of day for end_date
            end_date = end_date.replace(hour=23, minute=59, second=59)
            
            # Format for Google Calendar API
            start_time_str = start_date.isoformat() + 'Z'
            end_time_str = end_date.isoformat() + 'Z'
            
            # Call the Calendar API
            events_result = self.service.events().list(
                calendarId='primary',
                timeMin=start_time_str,
                timeMax=end_time_str,
                maxResults=args.max_results,
                singleEvents=True,
                orderBy='startTime'
            ).execute()
            
            events = events_result.get('items', [])
            
            # Format events for return
            formatted_events = []
            for event in events:
                start = event['start'].get('dateTime', event['start'].get('date'))
                end = event['end'].get('dateTime', event['end'].get('date'))
                
                # Extract attendees
                attendees = []
                if 'attendees' in event:
                    attendees = [attendee.get('email', '') for attendee in event['attendees']]
                
                formatted_event = CalendarEvent(
                    event_id=event.get('id', ''),
                    summary=event.get('summary', 'No Title'),
                    start_time=start,
                    end_time=end,
                    location=event.get('location', ''),
                    description=event.get('description', ''),
                    attendees=attendees
                )
                formatted_events.append(formatted_event)
            
            return {
                'events': formatted_events,
                'total_count': len(formatted_events)
            }
            
        except HttpError as error:
            print(f"An error occurred while getting events: {error}")
            raise
        except ValueError as error:
            print(f"Date parsing error: {error}")
            raise

    def edit_event(self, args: EditEventInput) -> Dict[str, str]:
        """Edit an existing calendar event."""
        try:
            # First, get the existing event
            event = self.service.events().get(calendarId='primary', eventId=args.event_id).execute()
            
            # Update fields only if new values are provided
            if args.summary:
                event['summary'] = args.summary
            
            if args.location:
                event['location'] = args.location
                
            if args.description:
                event['description'] = args.description
            
            # Handle date/time updates
            if args.start_date and args.start_time:
                datetime_str = f"{args.start_date} {args.start_time}"
                start_time = datetime.datetime.strptime(datetime_str, "%Y-%m-%d %H:%M")
                
                # Calculate end time
                if args.duration_minutes > 0:
                    end_time = start_time + datetime.timedelta(minutes=args.duration_minutes)
                else:
                    # Keep the original duration if no new duration specified
                    original_start = event['start'].get('dateTime')
                    original_end = event['end'].get('dateTime')
                    if original_start and original_end:
                        original_start_dt = datetime.datetime.fromisoformat(original_start.replace('Z', '+00:00'))
                        original_end_dt = datetime.datetime.fromisoformat(original_end.replace('Z', '+00:00'))
                        original_duration = original_end_dt - original_start_dt
                        end_time = start_time + original_duration
                    else:
                        # Default to 30 minutes if we can't determine original duration
                        end_time = start_time + datetime.timedelta(minutes=30)
                
                # Update the event times
                event['start'] = {
                    'dateTime': start_time.isoformat(),
                    'timeZone': 'America/Chicago',
                }
                event['end'] = {
                    'dateTime': end_time.isoformat(),
                    'timeZone': 'America/Chicago',
                }
            
            # Update the event
            updated_event = self.service.events().update(
                calendarId='primary', 
                eventId=args.event_id, 
                body=event
            ).execute()
            
            return {
                'success': True,
                'message': 'Event updated successfully',
                'event_link': updated_event.get('htmlLink', '')
            }
            
        except HttpError as error:
            error_msg = f"An error occurred while editing event: {error}"
            print(error_msg)
            return {
                'success': False,
                'message': error_msg,
                'event_link': ''
            }
        except ValueError as error:
            error_msg = f"Date parsing error: {error}"
            print(error_msg)
            return {
                'success': False,
                'message': error_msg,
                'event_link': ''
            }

    def create_placeholder_event(self, args: CreatePlaceholderEventInput) -> Dict[str, Any]:
        """Create a placeholder calendar event with TBD details."""
        try:
            # Parse date and time into datetime
            datetime_str = f"{args.date} {args.time}"
            start_time = datetime.datetime.strptime(datetime_str, "%Y-%m-%d %H:%M")
            end_time = start_time + datetime.timedelta(minutes=args.duration_minutes)
            
            # Format datetime for Google Calendar API
            start_time_str = start_time.isoformat()
            end_time_str = end_time.isoformat()
            
            # Create event title with placeholder indicator
            title = f"[PLACEHOLDER] {args.title}"
            
            # Create description with placeholder information
            placeholder_description = f"""
🔄 This is a PLACEHOLDER event created by AI Assistant

📋 DETAILS TO CONFIRM:
• Recipients: {args.recipient_placeholder}
• Location: {args.location}
• Final time: {args.time} ({args.duration_minutes} min)

⚠️ Please confirm or modify these details!

Original Description: {args.description}
            """.strip()
            
            # Create event body
            event_body = {
                'summary': title,
                'location': args.location,
                'description': placeholder_description,
                'start': {
                    'dateTime': start_time_str,
                    'timeZone': 'America/Chicago',
                },
                'end': {
                    'dateTime': end_time_str,
                    'timeZone': 'America/Chicago',
                },
                'colorId': '5'  # Yellow color to indicate placeholder
            }
            
            # Create the event
            event = self.service.events().insert(calendarId='primary', body=event_body).execute()
            
            placeholder_details = {
                'recipient_needs_confirmation': args.recipient_placeholder,
                'location_needs_confirmation': args.location,
                'time_needs_confirmation': f"{args.date} {args.time}",
                'duration_needs_confirmation': f"{args.duration_minutes} minutes"
            }
            
            return {
                'event_id': event.get('id'),
                'event_link': event.get('htmlLink'),
                'placeholder_details': placeholder_details
            }
            
        except HttpError as error:
            print(f"An error occurred creating placeholder: {error}")
            raise
        except ValueError as error:
            print(f"Date parsing error: {error}")
            raise

    def create_multiple_events(self, args: MultipleEventsInput) -> Dict[str, Any]:
        """Create multiple calendar events in a batch operation."""
        created_events = []
        failed_events = []
        
        print(f"Creating {len(args.events)} events in batch: {args.batch_description}")
        
        for i, event_data in enumerate(args.events):
            try:
                print(f"Processing event {i+1}/{len(args.events)}: {event_data.summary}")
                
                # Create the event using the existing single event creation method
                event_result = self.create_event(event_data)
                
                created_events.append(ScheduleMeetingOutput(
                    event_id=event_result['event_id'],
                    event_link=event_result['event_link']
                ))
                
                print(f"✅ Successfully created event {i+1}: {event_data.summary}")
                
            except Exception as error:
                error_details = {
                    'event_index': i,
                    'event_summary': event_data.summary,
                    'event_date': event_data.date,
                    'event_time': event_data.time,
                    'error_message': str(error),
                    'error_type': type(error).__name__
                }
                failed_events.append(error_details)
                print(f"Failed to create event {i+1}: {event_data.summary} - {str(error)}")
        
        total_created = len(created_events)
        total_failed = len(failed_events)
        
        # Create batch summary
        batch_summary = f"Batch operation completed: {total_created} events created successfully"
        if total_failed > 0:
            batch_summary += f", {total_failed} events failed"
        batch_summary += f" out of {len(args.events)} total events."
        
        print(f"📊 Batch Summary: {batch_summary}")
        
        return {
            'created_events': created_events,
            'failed_events': failed_events,
            'total_created': total_created,
            'total_failed': total_failed,
            'batch_summary': batch_summary
        }


class ScheduleMeetingTool(BaseTool[ScheduleMeetingInput, ScheduleMeetingOutput]):
    def __init__(self):
        super().__init__(
            ScheduleMeetingInput,
            ScheduleMeetingOutput,
            "schedule_meeting",
            "Schedule a meeting with a recipient at a specific date and time",
        )
        self.calendar_service = GoogleCalendarService()

    async def run(self, args: ScheduleMeetingInput, cancellation_token: CancellationToken) -> ScheduleMeetingOutput:
        print(f"Scheduling meeting with {args.recipient} on {args.date} at {args.time}")
        
        # Create the event in Google Calendar
        event_info = self.calendar_service.create_event(args)
        
        print(f"Meeting scheduled successfully. Event ID: {event_info['event_id']}")
        print(f"Event link: {event_info['event_link']}")
        
        return ScheduleMeetingOutput(
            event_id=event_info['event_id'],
            event_link=event_info['event_link']
        )


class GetEventsTool(BaseTool[GetEventsInput, GetEventsOutput]):
    def __init__(self):
        super().__init__(
            GetEventsInput,
            GetEventsOutput,
            "get_events",
            "Get calendar events for a specific date range",
        )
        self.calendar_service = GoogleCalendarService()

    async def run(self, args: GetEventsInput, cancellation_token: CancellationToken) -> GetEventsOutput:
        print(f"Getting events from {args.start_date} to {args.end_date}")
        
        # Get events from Google Calendar
        events_info = self.calendar_service.get_events(args)
        
        print(f"Found {events_info['total_count']} events")
        print(f"Standard event info log")
        print(f"Events: {events_info}")
        print(f"Events: {events_info['events']}")
        return GetEventsOutput(
            events=events_info['events'],
            total_count=events_info['total_count']
        )


class GetEventsApproxTool(BaseTool[GetEventsInput, GetEventsOutput]):
    def __init__(self):
        super().__init__(
            GetEventsInput,
            GetEventsOutput,
            "get_events_approx",
            "Get calendar events for the previous 7 days and the next 7 days",
        )
        self.calendar_service = GoogleCalendarService()

    async def run(self, args: GetEventsInput, cancellation_token: CancellationToken) -> GetEventsOutput:
        print(f"Getting events from {args.start_date} to {args.end_date}")
        
        # Get events from Google Calendar
        events_info = self.calendar_service.get_events_approx(args)
        
        print(f"Found {events_info['total_count']} events")
        
        return GetEventsOutput(
            events=events_info['events'],
            total_count=events_info['total_count']
        )


class EditEventTool(BaseTool[EditEventInput, EditEventOutput]):
    def __init__(self):
        super().__init__(
            EditEventInput,
            EditEventOutput,
            "edit_event",
            "Edit an existing calendar event",
        )
        self.calendar_service = GoogleCalendarService()

    async def run(self, args: EditEventInput, cancellation_token: CancellationToken) -> EditEventOutput:
        print(f"Editing event with ID: {args.event_id}")
        
        # Edit the event in Google Calendar
        edit_result = self.calendar_service.edit_event(args)
        
        if edit_result['success']:
            print(f"Event edited successfully")
        else:
            print(f"Failed to edit event: {edit_result['message']}")
        
        return EditEventOutput(
            success=edit_result['success'],
            message=edit_result['message'],
            event_link=edit_result['event_link']
        )


class CreatePlaceholderEventTool(BaseTool[CreatePlaceholderEventInput, CreatePlaceholderEventOutput]):
    def __init__(self):
        super().__init__(
            CreatePlaceholderEventInput,
            CreatePlaceholderEventOutput,
            "create_placeholder_event",
            "Create a placeholder calendar event with TBD details that need user confirmation",
        )
        self.calendar_service = GoogleCalendarService()

    async def run(self, args: CreatePlaceholderEventInput, cancellation_token: CancellationToken) -> CreatePlaceholderEventOutput:
        print(f"Creating placeholder event: {args.title} on {args.date} at {args.time}")
        
        # Create the placeholder event in Google Calendar
        placeholder_result = self.calendar_service.create_placeholder_event(args)
        
        print(f"Placeholder event created successfully. Event ID: {placeholder_result['event_id']}")
        
        return CreatePlaceholderEventOutput(
            event_id=placeholder_result['event_id'],
            event_link=placeholder_result['event_link'],
            placeholder_details=placeholder_result['placeholder_details']
        )


class MultipleEventsTool(BaseTool[MultipleEventsInput, MultipleEventsOutput]):
    def __init__(self):
        super().__init__(
            MultipleEventsInput,
            MultipleEventsOutput,
            "create_multiple_events",
            "Create multiple calendar events in a single batch operation",
        )
        self.calendar_service = GoogleCalendarService()

    async def run(self, args: MultipleEventsInput, cancellation_token: CancellationToken) -> MultipleEventsOutput:
        print(f"Creating multiple events: {args.batch_description}")
        print(f"Total events to create: {len(args.events)}")
        
        # Create the events in Google Calendar
        batch_result = self.calendar_service.create_multiple_events(args)
        
        print(f"Batch operation completed: {batch_result['total_created']} created, {batch_result['total_failed']} failed")
        
        return MultipleEventsOutput(
            created_events=batch_result['created_events'],
            failed_events=batch_result['failed_events'],
            total_created=batch_result['total_created'],
            total_failed=batch_result['total_failed'],
            batch_summary=batch_result['batch_summary']
        )


@type_subscription("scheduling_assistant_conversation")
class SchedulingAssistantAgent(RoutedAgent):
    def __init__(
        self,
        name: str,
        description: str,
        model_client: ChatCompletionClient,
        initial_message: AssistantTextMessage | None = None,
    ) -> None:
        super().__init__(description)
        self._model_context = BufferedChatCompletionContext(
            buffer_size=5,
            initial_messages=[UserMessage(content=initial_message.content, source=initial_message.source)]
            if initial_message
            else None,
        )
        self._name = name
        self._model_client = model_client
        self._system_messages = [
            SystemMessage(
                content=f"""
🗓️ **I am a helpful AI assistant that helps with calendar management.**

## 🛠️ **My Capabilities:**
1. 📅 **Schedule new meetings** and add them directly to your Google Calendar
2. 👀 **View existing events** in your calendar for a specific date range
3. ✏️ **Edit existing calendar events** (modify time, location, description, etc.)
4. 📝 **Create placeholder events** with default values when information is incomplete
5. 🔄 **Create multiple events** in a single batch operation for efficiency

## 📋 **Information I Need:**
- **For scheduling:** date, time, recipient, and optionally duration, summary, description, location, and attendee email
- **For viewing events:** date range (start and end dates) or I will get events for the previous 7 days and the next 7 days by default
- **For editing events:** event ID and the fields you want to change
- **For placeholder events:** I can create events with reasonable defaults and mark them for later confirmation
- **For multiple events:** I can create several events at once when provided with a list of event details

## 🔄 **Multiple Events Creation:**
When users request multiple meetings or events (e.g., "schedule 3 meetings", "create events for the week", "batch schedule"), I will use the **multiple events tool** to create them all at once. This is more efficient than creating individual events and provides a comprehensive summary of the batch operation.

### 📤 **Expected Format from Orchestrator:**
I expect the orchestrator to provide complete event details in the following format for multiple events:
- ✅ A **list of events** with all necessary details (date, time, duration, recipient, summary, etc.)
- 📄 A **batch description** explaining the purpose of these events
- 🎯 **All events should be fully specified** to avoid the need for additional clarification

## 🔖 **Placeholder Events Guidelines:**
When creating placeholder events:
- ⏰ I'll use **working hours (9 AM - 6 PM)** for default times
- ⏱️ I'll set **reasonable durations (30-60 minutes)**
- 👥 I'll mark recipients and locations as **"TBD"** if not specified
- ⚠️ The event will be **clearly marked as a placeholder** needing confirmation

## 📝 **Response Format:**
I will always respond using **markdown formatting** with:
- 📊 **Clear headers and sections**
- ✅ **Success indicators** with green checkmarks
- ❌ **Error indicators** with red X marks
- 🔗 **Clickable links** to calendar events
- 📈 **Organized lists** and bullet points
- 🎯 **Emojis** for visual clarity and engagement

If there are missing parameters, I will ask for them or create reasonable placeholders.

📅 **Today's date is {datetime.datetime.now().strftime("%Y-%m-%d")}**
"""
            )
        ]

    @message_handler
    async def handle_message(self, message: UserTextMessage, ctx: MessageContext) -> None:
        await self._model_context.add_message(UserMessage(content=message.content, source=message.source))

        tools = [ScheduleMeetingTool(), GetEventsTool(), GetEventsApproxTool(), EditEventTool(), CreatePlaceholderEventTool(), MultipleEventsTool()]
        response = await self._model_client.create(
            self._system_messages + (await self._model_context.get_messages()), tools=tools
        )

        if isinstance(response.content, list) and all(isinstance(item, FunctionCall) for item in response.content):
            for call in response.content:
                tool = next((tool for tool in tools if tool.name == call.name), None)
                if tool is None:
                    raise ValueError(f"Tool not found: {call.name}")
                arguments = json.loads(call.arguments)
                result = await tool.run_json(arguments, ctx.cancellation_token)
                
                # Handle different tool responses
                if call.name == "schedule_meeting":
                    response_text = f"✅ **Meeting scheduled successfully!** 📅\n\n🔗 **View your event:** {result.event_link}"
                elif call.name == "get_events" or call.name == "get_events_approx":
                    if result.total_count == 0:
                        response_text = f"📭 **No events found** for the specified date range."
                    else:
                        events_list = []
                        for event in result.events:
                            attendees_str = ", ".join(event.attendees) if event.attendees else "None"
                            events_list.append(
                                f"📋 **{event.summary}**\n"
                                f"   📅 **When:** {event.start_time} - {event.end_time}\n"
                                f"   📍 **Location:** {event.location or 'No location'}\n"
                                f"   👥 **Attendees:** {attendees_str}\n"
                                f"   🆔 **Event ID:** `{event.event_id}`\n"
                                f"   📝 **Description:** {event.description or 'No description'}"
                            )
                        response_text = f"📊 **Found {result.total_count} events:**\n\n" + "\n\n".join(events_list)
                elif call.name == "edit_event":
                    if result.success:
                        response_text = f"✅ **Event updated successfully!** {result.message} 🎉"
                        if result.event_link:
                            response_text += f"\n\n🔗 **View updated event:** {result.event_link}"
                    else:
                        response_text = f"❌ **Failed to update event:** {result.message}"
                elif call.name == "create_placeholder_event":
                    response_text = f"🔖 **Placeholder event created successfully!** \n\n📋 **Event ID:** `{result.event_id}`"
                    if result.event_link:
                        response_text += f"\n🔗 **View event:** {result.event_link}"
                    response_text += f"\n\n⚠️ **Note:** This is a placeholder event that needs confirmation!"
                elif call.name == "create_multiple_events":
                    response_text = f"## 🔄 **Multiple Events Creation Summary**\n\n"
                    response_text += f"📊 **Results:** {result.total_created} events created successfully"
                    if result.total_failed > 0:
                        response_text += f", {result.total_failed} events failed"
                    response_text += f"\n\n"
                    
                    if result.created_events:
                        response_text += f"### ✅ **Successfully Created Events:**\n"
                        for i, event in enumerate(result.created_events, 1):
                            response_text += f"{i}. 📋 **Event ID:** `{event.event_id}`\n   🔗 **Link:** {event.event_link}\n\n"
                    
                    if result.failed_events:
                        response_text += f"### ❌ **Failed Events:**\n"
                        for i, failed in enumerate(result.failed_events, 1):
                            response_text += f"{i}. 📅 **{failed.get('event_summary', 'Unknown')}** on {failed.get('event_date', 'Unknown')} at {failed.get('event_time', 'Unknown')}\n"
                            response_text += f"   ⚠️ **Error:** {failed.get('error_message', 'Unknown error')}\n\n"
                    
                    response_text += f"📝 **Summary:** {result.batch_summary}"
                else:
                    response_text = f"✅ **Tool {call.name} executed successfully** 🎉"
                
                speech = AssistantTextMessage(content=response_text, source=self.metadata["type"])
                await self._model_context.add_message(AssistantMessage(content=response_text, source=self.metadata["type"]))
                await self.publish_message(speech, topic_id=DefaultTopicId("scheduling_assistant_conversation"))
                
            await self.publish_message(
                TerminateMessage(content="Calendar operation completed"),
                topic_id=DefaultTopicId("scheduling_assistant_conversation"),
            )
            return

        assert isinstance(response.content, str)
        speech = AssistantTextMessage(content=response.content, source=self.metadata["type"])
        await self._model_context.add_message(AssistantMessage(content=response.content, source=self.metadata["type"]))

        await self.publish_message(speech, topic_id=DefaultTopicId("scheduling_assistant_conversation"))

    async def save_state(self) -> Mapping[str, Any]:
        return {
            "memory": await self._model_context.save_state(),
        }

    async def load_state(self, state: Mapping[str, Any]) -> None:
        await self._model_context.load_state(state["memory"])


class NeedsUserInputHandler(DefaultInterventionHandler):
    def __init__(self):
        self.question_for_user: GetSlowUserMessage | None = None

    async def on_publish(self, message: Any, *, message_context: MessageContext) -> Any:
        if isinstance(message, GetSlowUserMessage):
            self.question_for_user = message
        return message

    @property
    def needs_user_input(self) -> bool:
        return self.question_for_user is not None

    @property
    def user_input_content(self) -> str | None:
        if self.question_for_user is None:
            return None
        return self.question_for_user.content


class TerminationHandler(DefaultInterventionHandler):
    def __init__(self):
        self.terminateMessage: TerminateMessage | None = None

    async def on_publish(self, message: Any, *, message_context: MessageContext) -> Any:
        if isinstance(message, TerminateMessage):
            self.terminateMessage = message
        return message

    @property
    def is_terminated(self) -> bool:
        return self.terminateMessage is not None

    @property
    def termination_msg(self) -> str | None:
        if self.terminateMessage is None:
            return None
        return self.terminateMessage.content


async def run_calendar_agent(model_config: Dict[str, Any], latest_user_input: Optional[str] = None) -> str:
    print("Running calendar agent")
    result = await main(model_config, latest_user_input)
    
    # Always return a string - if result is None, it means the operation was completed
    if result is None:
        return "Calendar operation completed successfully."
    return result

async def main(model_config: Dict[str, Any], latest_user_input: Optional[str] = None) -> None | str:
    """
    Asynchronous function that serves as the entry point of the program.
    This function initializes the necessary components for the program and registers the user and scheduling assistant agents.
    If a user input is provided, it loads the state (from some persistent layer) and publishes the user input message to
    the scheduling assistant. Otherwise, it adds an initial message to the scheduling assistant's history and publishes it
    to the message queue. The program then starts running and stops when either the termination handler is triggered
    or user input is needed. Finally, it saves the state and returns the user input needed if any.

    Args:
        latest_user_input (Optional[str]): The latest user input. Defaults to None.

    Returns:
        None or str: The user input needed if the program requires user input, otherwise None.
    """
    global state_persister

    model_client = ChatCompletionClient.load_component(model_config)

    termination_handler = TerminationHandler()
    needs_user_input_handler = NeedsUserInputHandler()
    runtime = SingleThreadedAgentRuntime(intervention_handlers=[needs_user_input_handler, termination_handler])

    await SlowUserProxyAgent.register(runtime, "User", lambda: SlowUserProxyAgent("User", "I am a user"))

    initial_schedule_assistant_message = AssistantTextMessage(
        content="Hi! How can I help you? I can manage your Google Calendar - schedule new meetings, view existing events, and edit calendar events.", source="User"
    )
    await SchedulingAssistantAgent.register(
        runtime,
        "SchedulingAssistant",
        lambda: SchedulingAssistantAgent(
            "SchedulingAssistant",
            description="AI that helps you schedule meetings and adds them to Google Calendar",
            model_client=model_client,
            initial_message=initial_schedule_assistant_message,
        ),
    )

    runtime_initiation_message: UserTextMessage | AssistantTextMessage
    if latest_user_input is not None:
        runtime_initiation_message = UserTextMessage(content=latest_user_input, source="User")
    else:
        runtime_initiation_message = initial_schedule_assistant_message
    state = state_persister.load_content()

    if state:
        await runtime.load_state(state)
    await runtime.publish_message(
        runtime_initiation_message,
        DefaultTopicId("scheduling_assistant_conversation"),
    )

    runtime.start()
    await runtime.stop_when(lambda: termination_handler.is_terminated or needs_user_input_handler.needs_user_input)
    await model_client.close()

    user_input_needed = None
    if needs_user_input_handler.user_input_content is not None:
        user_input_needed = needs_user_input_handler.user_input_content
    elif termination_handler.is_terminated:
        print("Terminated - ", termination_handler.termination_msg)

    state_to_persist = await runtime.save_state()
    state_persister.save_content(state_to_persist)

    return user_input_needed


async def ainput(prompt: str = "") -> str:
    with ThreadPoolExecutor(1, "AsyncInput") as executor:
        return await asyncio.get_event_loop().run_in_executor(executor, input, prompt)


if __name__ == "__main__":
    # import logging

    # logging.basicConfig(level=logging.WARNING)
    # logging.getLogger("autogen_core").setLevel(logging.DEBUG)

    # if os.path.exists("state.json"):
    #     os.remove("state.json")

    with open("model_config.yml") as f:
        model_config = yaml.safe_load(f)

    def get_user_input(question_for_user: str):
        print("--------------------------QUESTION_FOR_USER--------------------------")
        print(question_for_user)
        print("---------------------------------------------------------------------")
        user_input = input("Enter your input: ")
        return user_input

    async def run_main(question_for_user: str | None = None):
        if question_for_user:
            user_input = get_user_input(question_for_user)
        else:
            user_input = None
        user_input_needed = await main(model_config, user_input)
        if user_input_needed:
            await run_main(user_input_needed)

    asyncio.run(run_main())
