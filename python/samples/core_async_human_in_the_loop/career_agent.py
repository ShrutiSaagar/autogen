# career_agent.py

import datetime
import json
from calendar_agent import ScheduleMeetingInput, ScheduleMeetingTool

class CareerAgent:
    def __init__(self, user_data_path='user_career_data.json'):
        self.user_data_path = user_data_path
        self.tasks = []
        self.tool = ScheduleMeetingTool()
        self.load_user_data()

    def load_user_data(self):
        with open(self.user_data_path, 'r') as f:
            self.user_data = json.load(f)

    def suggest_career_tasks(self):
        """
        Generate career-related tasks and schedule them using ScheduleMeetingTool.
        """
        career_goals = self.user_data.get("career_goals", [])
        base_date = datetime.datetime.now()

        for i, goal in enumerate(career_goals):
            start_time = base_date + datetime.timedelta(days=i, hours=10)
            end_time = start_time + datetime.timedelta(hours=1)

            input_data = ScheduleMeetingInput(
                recipient="Career Planning",
                date=start_time.strftime("%Y-%m-%d"),
                time=start_time.strftime("%H:%M"),
                duration_minutes=60,
                summary=goal.get("title", "Career Task"),
                description=goal.get("description", ""),
                location="",
                attendee_email=""
            )

            event_output = self.tool.calendar_service.create_event(input_data)

            self.tasks.append({
                "summary": input_data.summary,
                "description": input_data.description,
                "start": start_time.isoformat(),
                "end": end_time.isoformat(),
                "event_link": event_output.get("event_link", "")
            })

    def schedule_recommendation_requests(self):
        """
        Schedule reminder to request recommendation letters before deadline.
        """
        recs = self.user_data.get("recommendation_requests", [])
        for item in recs:
            deadline = datetime.datetime.strptime(item["deadline"], "%Y-%m-%d")
            remind_date = deadline - datetime.timedelta(days=7)

            input_data = ScheduleMeetingInput(
                recipient="Recommendation Reminder",
                date=remind_date.strftime("%Y-%m-%d"),
                time="09:00",
                summary=f"Email {item['professor']} for LOR",
                description=f"Request LOR for {item['program']} program.",
                duration_minutes=30,
                location="",
                attendee_email=""
            )

            event = self.tool.calendar_service.create_event(input_data)
            self.tasks.append({
                "summary": input_data.summary,
                "description": input_data.description,
                "start": remind_date.isoformat(),
                "end": (remind_date + datetime.timedelta(minutes=30)).isoformat(),
                "event_link": event["event_link"]
            })

    def schedule_job_deadline_reminders(self):
        """
        Schedule alerts 2 days before job/internship application deadlines.
        """
        jobs = self.user_data.get("job_deadlines", [])
        for job in jobs:
            deadline = datetime.datetime.strptime(job["deadline"], "%Y-%m-%d")
            reminder_date = deadline - datetime.timedelta(days=2)

            input_data = ScheduleMeetingInput(
                recipient="Job Reminder",
                date=reminder_date.strftime("%Y-%m-%d"),
                time="10:00",
                summary=f"Submit application: {job['company']} - {job['position']}",
                description="Final review and submission.",
                duration_minutes=45,
                location="",
                attendee_email=""
            )

            event = self.tool.calendar_service.create_event(input_data)
            self.tasks.append({
                "summary": input_data.summary,
                "description": input_data.description,
                "start": reminder_date.isoformat(),
                "end": (reminder_date + datetime.timedelta(minutes=45)).isoformat(),
                "event_link": event["event_link"]
            })

    def schedule_skill_learning(self):
        """
        Schedule recurring skill development sessions.
        """
        skills = self.user_data.get("skill_development", [])
        start_date = datetime.datetime.now()

        for skill in skills:
            for i in range(3):  # Schedule 3 future sessions
                session_day = start_date + datetime.timedelta(days=i * skill["frequency_days"])

                input_data = ScheduleMeetingInput(
                    recipient="Skill Learning",
                    date=session_day.strftime("%Y-%m-%d"),
                    time="19:00",
                    summary=f"{skill['platform']}: Learn {skill['skill']}",
                    description=f"Skill-building session for {skill['skill']}.",
                    duration_minutes=60,
                    location="",
                    attendee_email=""
                )

                event = self.tool.calendar_service.create_event(input_data)
                self.tasks.append({
                    "summary": input_data.summary,
                    "description": input_data.description,
                    "start": session_day.isoformat(),
                    "end": (session_day + datetime.timedelta(minutes=60)).isoformat(),
                    "event_link": event["event_link"]
                })

    def respond_to_schedule_change(self, delay_minutes):
        """
        Adjust task times locally by a given delay.
        """
        for task in self.tasks:
            start = datetime.datetime.fromisoformat(task["start"]) + datetime.timedelta(minutes=delay_minutes)
            end = datetime.datetime.fromisoformat(task["end"]) + datetime.timedelta(minutes=delay_minutes)
            task["start"] = start.isoformat()
            task["end"] = end.isoformat()

    def get_tasks(self):
        return self.tasks

    def __str__(self):
        return f"CareerAgent managing {len(self.tasks)} tasks."
