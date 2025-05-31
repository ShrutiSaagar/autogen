# test_career_agent.py

from career_agent import CareerAgent

def main():
    agent = CareerAgent()
    agent.suggest_career_tasks()
    agent.schedule_recommendation_requests()
    agent.schedule_job_deadline_reminders()
    agent.schedule_skill_learning()

    print("\nAll Scheduled Tasks:")
    for task in agent.get_tasks():
        print(f"📌 {task['summary']}")
        print(f"🕒 {task['start']} to {task['end']}")
        print(f"🔗 {task['event_link']}\n")

if __name__ == "__main__":
    main()
