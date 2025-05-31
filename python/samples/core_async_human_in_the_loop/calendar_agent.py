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
I am a helpful AI assistant that helps schedule meetings.
I can add events directly to your Google Calendar.
I will ask for necessary details like date, time, recipient, and optionally duration, summary, description, location, and attendee email.
If there are missing parameters, I will ask for them.

Today's date is {datetime.datetime.now().strftime("%Y-%m-%d")}
"""
            )
        ]

    @message_handler
    async def handle_message(self, message: UserTextMessage, ctx: MessageContext) -> None:
        await self._model_context.add_message(UserMessage(content=message.content, source=message.source))

        tools = [ScheduleMeetingTool()]
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
                
                # Craft a response message about the scheduled meeting
                response_text = f"Meeting scheduled successfully! You can view it here: {result.event_link}"
                
                speech = AssistantTextMessage(content=response_text, source=self.metadata["type"])
                await self._model_context.add_message(AssistantMessage(content=response_text, source=self.metadata["type"]))
                await self.publish_message(speech, topic_id=DefaultTopicId("scheduling_assistant_conversation"))
                
            await self.publish_message(
                TerminateMessage(content="Meeting scheduled and added to Google Calendar"),
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


async def run_calendar_agent(model_config: Dict[str, Any], latest_user_input: Optional[str] = None) -> None | str:
    print("Running calendar agent")
    return await main(model_config, latest_user_input)

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
        content="Hi! How can I help you? I can schedule meetings and add them directly to your Google Calendar.", source="User"
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
