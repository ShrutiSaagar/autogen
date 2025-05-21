#!/usr/bin/env python3
# calendar_team.py - A multi-agent system for web research and calendar management

import asyncio
import json
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

# Try to load environment variables from .env file
try:
    from dotenv import load_dotenv
    # Look for .env file, fallback to env_example if not found
    if os.path.exists(os.path.join(os.path.dirname(__file__), ".env")):
        load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))
    else:
        load_dotenv(os.path.join(os.path.dirname(__file__), "env_example"))
    print("Environment variables loaded from .env file")
except ImportError:
    print("python-dotenv not installed. Using environment variables from system.")

# Fix for Windows to avoid issues with subprocesses
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

# Import AutoGen components
from autogen_agentchat.agents import AssistantAgent, UserProxyAgent
from autogen_agentchat.conditions import TextMentionTermination
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.ui import Console
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_ext.agents.web_surfer import MultimodalWebSurfer

# Import our calendar updater
from calendar_updater import CalendarUpdater

# Constants
SCHEDULE_FILE = Path(__file__).parent / "dummy_schedule.json"
CALENDAR_OUTPUT_FILE = Path(__file__).parent / "updated_calendar.json"

class CalendarOrchestrationWorkflow:
    """The main calendar orchestration workflow."""
    
    def __init__(self):
        self.schedule_file = SCHEDULE_FILE
        self.output_file = CALENDAR_OUTPUT_FILE
        self.calendar_updater = CalendarUpdater(SCHEDULE_FILE, CALENDAR_OUTPUT_FILE)
    
    async def run(self):
        """Run the calendar orchestration workflow."""
        # Initialize the model client with OpenAI API
        model_client = OpenAIChatCompletionClient(
            model="gpt-4o",  # Using GPT-4o for best reasoning capabilities
        )
        
        # Create the WebResearchAgent - responsible for web browsing and information gathering
        web_research_agent = MultimodalWebSurfer(
            name="web_researcher",
            model_client=model_client,
            headless=False,  # Set to True in production
            animate_actions=True,  # Visual feedback of agent actions
            description="""You are a web research specialist. Your role is to find relevant information online 
            based on user requests and events. You search for details about events, locations, travel time, and 
            other contextual information needed for scheduling. When finished with your research, summarize your 
            findings concisely for the team."""
        )
        
        # Create the ScheduleAnalyzerAgent - analyzes existing schedule and identifies constraints
        schedule_analyzer = AssistantAgent(
            name="schedule_analyzer",
            model_client=model_client,
            description="Schedule analysis expert",
            system_message="""You are a schedule analysis expert. You analyze the user's existing schedule to identify:
            1. Available time slots
            2. Scheduling constraints
            3. Conflicts between proposed events
            4. Priority assessments
            
            Your job is to read the schedule data and provide analytical insights. You do not modify the schedule directly.
            Always consider user preferences like work hours, lunch time, and focus time when analyzing availability.
            
            When asked to find available slots, respond with specific time ranges that are open in the schedule."""
        )
        
        # Create the CalendarManagerAgent - responsible for updating the calendar
        calendar_manager = AssistantAgent(
            name="calendar_manager",
            model_client=model_client,
            description="Calendar management specialist",
            system_message="""You are a calendar management specialist. Your job is to:
            1. Create new calendar entries based on user requests and team recommendations
            2. Resolve scheduling conflicts
            3. Optimize the calendar based on priorities
            4. Format calendar entries correctly
            
            You are the only agent authorized to make changes to the calendar data. When you finalize changes,
            format them as valid JSON that matches the expected schema for saving to the calendar file.
            
            The schema for events is:
            {
              "id": "unique_id",
              "title": "Event Title",
              "description": "Event description",
              "start": "YYYY-MM-DDThh:mm:ss",
              "end": "YYYY-MM-DDThh:mm:ss",
              "location": "Location",
              "priority": "critical|high|medium|low"
            }
            
            When finalizing updates, format your response as follows:
            
            CALENDAR_UPDATES:
            {
              "action": "add|update|delete",
              "events": [
                {event object following the schema above}
              ]
            }"""
        )
        
        # Create the OrchestratorAgent - coordinates the entire workflow
        orchestrator = AssistantAgent(
            name="orchestrator",
            model_client=model_client,
            description="Team workflow orchestrator",
            system_message="""You are the orchestrator of the calendar planning team. Your job is to:
            1. Understand user requests
            2. Coordinate the work of other specialist agents
            3. Synthesize information from all agents
            4. Make final recommendations to the user
            5. Ensure the entire workflow is completed properly
            
            Start by analyzing the user's request, then determine which specialist agent should act next.
            When a request comes in, break it down into specific tasks for each agent:
            - For web_researcher: Ask to find relevant information about events, locations, travel times
            - For schedule_analyzer: Ask to analyze the schedule for availability and constraints
            - For calendar_manager: Task with creating, updating, or deleting events
            
            Track progress and ensure all necessary information is gathered before finalizing decisions.
            
            When the workflow is complete, provide a summary of the changes and ask calendar_manager to prepare the final calendar updates.
            When everything is complete, include "TASK_COMPLETE" in your final message."""
        )
        
        # Create UserProxyAgent to represent the human user
        user_proxy = UserProxyAgent(
            name="user_proxy",
            description="Human user requesting calendar management",
        )
        
        # Process calendar updates when detected in messages
        async def process_calendar_updates(message_content):
            """Process calendar updates from the CalendarManagerAgent."""
            if "CALENDAR_UPDATES:" in message_content:
                # Extract the JSON part
                try:
                    update_start = message_content.find("CALENDAR_UPDATES:") + len("CALENDAR_UPDATES:")
                    update_json = message_content[update_start:].strip()
                    update_data = json.loads(update_json)
                    
                    # Process the updates
                    action = update_data.get("action")
                    events = update_data.get("events", [])
                    
                    for event in events:
                        if action == "add":
                            self.calendar_updater.add_event(
                                title=event.get("title", "Untitled"),
                                description=event.get("description", ""),
                                start=event.get("start"),
                                end=event.get("end"),
                                location=event.get("location", ""),
                                priority=event.get("priority", "medium")
                            )
                            print(f"Added event: {event.get('title')}")
                        
                        elif action == "update" and "id" in event:
                            self.calendar_updater.update_event(
                                event_id=event["id"],
                                **{k: v for k, v in event.items() if k != "id"}
                            )
                            print(f"Updated event: {event.get('title')}")
                        
                        elif action == "delete" and "id" in event:
                            self.calendar_updater.delete_event(event["id"])
                            print(f"Deleted event: {event.get('title')}")
                    
                    # Save the calendar after processing updates
                    self.calendar_updater.save_calendar()
                    
                    return True
                except Exception as e:
                    print(f"Error processing calendar updates: {e}")
            
            return False
        
        # Add a message handler to the calendar_manager to process updates
        async def calendar_manager_handler(message, sender, recipient):
            """Handle messages from the calendar manager to process updates."""
            if sender == calendar_manager.name and recipient == orchestrator.name:
                content = message.get("content", "")
                if await process_calendar_updates(content):
                    # Append confirmation to the message
                    message["content"] += "\n\nCalendar has been updated successfully."
            return message
        
        # Register the handler with the orchestrator
        orchestrator.register_message_handler(calendar_manager_handler)
        
        # Load the current schedule into context
        current_schedule = self.calendar_updater.calendar_data
        schedule_context = f"""
        Current schedule and preferences: {json.dumps(current_schedule, indent=2)}
        
        Please analyze this schedule and work with the team to accommodate the user's new requests.
        """
        
        # Define the team of agents
        team_members = [
            orchestrator,
            web_research_agent,
            schedule_analyzer, 
            calendar_manager,
            user_proxy
        ]
        
        # Create termination condition
        termination = TextMentionTermination("TASK_COMPLETE")
        
        # Initialize the group chat with the orchestrator as the leader
        team_chat = RoundRobinGroupChat(
            agents=team_members,
            termination_condition=termination,
            select_speaker=lambda message, state: orchestrator.name  # Orchestrator always speaks first after user
        )
        
        try:
            # Start the system with initial context about the schedule
            print("\n===== Starting Calendar Planning Team =====\n")
            print("The system has loaded your current schedule and preferences.")
            print("You can now describe what you'd like to add or modify in your calendar.\n")
            
            # Start with the schedule context and let the user begin interaction
            initial_message = f"I need to review my schedule and make some adjustments. {schedule_context}"
            
            # Run the team chat
            await Console(team_chat.run_stream(task=initial_message))
            
            # After completion, the calendar_manager would have produced updated calendar data
            # In a real system, this would be saved and synchronized with an actual calendar service
            print("\n===== Calendar Planning Complete =====\n")
            print(f"Updated calendar saved to: {self.output_file}")
            
            # Get and display a summary of the updated calendar
            summary = self.calendar_updater.get_calendar_summary()
            print("\nCalendar Summary:")
            print(f"Total events: {summary['total_events']}")
            print("Events by priority:")
            for priority, count in summary["events_by_priority"].items():
                print(f"  - {priority.capitalize()}: {count}")
            print(f"Upcoming events (next 7 days): {len(summary['upcoming_events'])}")
            
        finally:
            # Cleanup resources
            await web_research_agent.close()
            await model_client.close()

async def main():
    """Main entry point for the calendar team."""
    workflow = CalendarOrchestrationWorkflow()
    await workflow.run()

if __name__ == "__main__":
    asyncio.run(main())