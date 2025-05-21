# Calendar Agent Team

A multi-agent system using AutoGen that helps manage your Google Calendar by using 4 specialized agents:

1. **OrchestratorAgent** - Coordinates the workflow and makes final recommendations
2. **WebResearchAgent** - Browses the web to gather relevant information
3. **ScheduleAnalyzerAgent** - Analyzes calendar for availability and conflicts
4. **CalendarManagerAgent** - Creates, updates, and deletes calendar events

## Setup

1. Clone this repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Install Playwright for web browsing:
   ```bash
   playwright install
   ```
4. Copy the environment template:
   ```bash
   cp env_example .env
   ```
5. Edit `.env` and add your OpenAI API key

## Usage

Run the multi-agent system:

```bash
python calendar_team.py
```

The system will:
1. Load your current schedule from `dummy_schedule.json`
2. Allow you to describe what calendar changes you need
3. Coordinate among the 4 agents to research, analyze, and plan your calendar
4. Update the calendar and save it to `updated_calendar.json`

## Examples of Tasks

- "I need to schedule a dentist appointment next Tuesday afternoon"
- "Find a good time for a team lunch next week, considering everyone's schedules"
- "I have to attend a 3-day conference in San Francisco next month. Look up flights and hotels and schedule travel time."
- "Move my weekly team meeting from Monday to Wednesday morning"
- "I need dedicated focus time for writing a report - find 2-hour blocks on 3 days this week"

## How It Works

1. **OrchestratorAgent** coordinates the entire workflow
2. **WebResearchAgent** searches for relevant details online (location, travel time, etc.)
3. **ScheduleAnalyzerAgent** analyzes your calendar for availability
4. **CalendarManagerAgent** makes the necessary updates to your calendar

## Customization

- Edit `dummy_schedule.json` to match your actual schedule
- Modify agent prompts in `calendar_team.py` to change agent behaviors
- Extend `calendar_updater.py` to add more calendar manipulation features 