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


@type_subscription("commute_agent_conversation")
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
            GetSlowUserMessage(content=message.content), topic_id=DefaultTopicId("commute_agent_conversation")
        )

    async def save_state(self) -> Mapping[str, Any]:
        state_to_save = {
            "memory": await self._model_context.save_state(),
        }
        return state_to_save

    async def load_state(self, state: Mapping[str, Any]) -> None:
        await self._model_context.load_state(state["memory"])


class CommuteAgentInput(BaseModel):
    current_location: str = Field(description="User's current location")
    destination: str = Field(description="Meeting destination")
    time: str = Field(description="Meeting time")


class CommuteAgentOutput(BaseModel):
    pass

import googlemaps
class CommuteAgentTool(BaseTool[CommuteAgentInput, CommuteAgentOutput]):
    def __init__(self, api_key: str):
        super().__init__(
            CommuteAgentInput,
            CommuteAgentOutput,
            "commute_assistant",
            "Calculate ETA and suggest navigation using Google Maps",
        )
        self.client = googlemaps.Client(key=api_key)

    async def run(self, args: CommuteAgentInput, cancellation_token: CancellationToken) -> CommuteAgentOutput:
        # Get directions
        directions_result = self.client.directions(
            args.current_location,
            args.destination,
            mode="driving",
            departure_time=datetime.now()
        )

        if directions_result:
            eta = directions_result[0]['legs'][0]['duration']['text']
            print(f"It will take approximately {eta} to reach {args.destination}.")
            map_url = f"https://www.google.com/maps/dir/?api=1&origin= {args.current_location}&destination={args.destination}"
            print(f"Would you like to open Google Maps to navigate? Open: {map_url}")
        else:
            print("Could not retrieve directions.")
        return CommuteAgentOutput()


@type_subscription("commute_agent_conversation")
class CommuteAssistantAgent(RoutedAgent):
    def __init__(
        self,
        name: str,
        description: str,
        model_client: ChatCompletionClient,
        initial_message: AssistantTextMessage | None = None,
        *,
    google_maps_api_key: str,
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
        self._google_maps_api_key = google_maps_api_key
        self._system_messages = [
            SystemMessage(
                content=f"""
I am a helpful AI assistant that helps schedule meetings.
If there are missing parameters, I will assume some options with the information I have and have been provided with and ask if its fine or should I give more options and ask for the final decision or direction on the search.
I can also provide an estimate of the time it will take to reach the destination based on the api response.
I can also provide a map url to navigate to the destination.
I can also provide a list of the best routes to reach the destination.

Today's date is {datetime.datetime.now().strftime("%Y-%m-%d")}
"""
            )
        ]

    @message_handler
    async def handle_message(self, message: UserTextMessage, ctx: MessageContext) -> None:
        await self._model_context.add_message(UserMessage(content=message.content, source=message.source))

        tools = [CommuteAgentTool(api_key=self._google_maps_api_key)]
        response = await self._model_client.create(
            self._system_messages + (await self._model_context.get_messages()), tools=tools
        )

        if isinstance(response.content, list) and all(isinstance(item, FunctionCall) for item in response.content):
            for call in response.content:
                tool = next((tool for tool in tools if tool.name == call.name), None)
                if tool is None:
                    raise ValueError(f"Tool not found: {call.name}")
                arguments = json.loads(call.arguments)
                await tool.run_json(arguments, ctx.cancellation_token)
            await self.publish_message(
                TerminateMessage(content="Meeting scheduled"),
                topic_id=DefaultTopicId("commute_agent_conversation"),
            )
            return

        assert isinstance(response.content, str)
        speech = AssistantTextMessage(content=response.content, source=self.metadata["type"])
        await self._model_context.add_message(AssistantMessage(content=response.content, source=self.metadata["type"]))

        await self.publish_message(speech, topic_id=DefaultTopicId("commute_agent_conversation"))

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
    print("--------------------------------MAIN Commute Agent--------------------------------")

    model_client = ChatCompletionClient.load_component(model_config)

    termination_handler = TerminationHandler()
    needs_user_input_handler = NeedsUserInputHandler()
    runtime = SingleThreadedAgentRuntime(intervention_handlers=[needs_user_input_handler, termination_handler])

    await SlowUserProxyAgent.register(runtime, "User", lambda: SlowUserProxyAgent("User", "I am a user"))

    initial_schedule_assistant_message = AssistantTextMessage(
            content="Hi! I can help you find the fastest route. What's your current location?", source="User"
    )
    
    google_maps_api_key = model_config.get("google_maps_api_key")

    await CommuteAssistantAgent.register(
    runtime,
    "SchedulingAssistant",
    lambda: CommuteAssistantAgent(
        "SchedulingAssistant",
        description="AI that helps you schedule meetings",
        model_client=model_client,
        initial_message=initial_schedule_assistant_message,
        google_maps_api_key=google_maps_api_key,
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
        DefaultTopicId("commute_agent_conversation"),
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
    import os
    import webbrowser  # To open maps link automatically
    import googlemaps  # For Directions API

    with open("model_config.yml") as f:
        model_config = yaml.safe_load(f)

    # Load Google Maps API key from config
    gmaps_api_key = model_config.get("google_maps_api_key")
    if not gmaps_api_key:
        raise ValueError("Google Maps API key is missing in model_config.yml")

    gmaps_client = googlemaps.Client(key=gmaps_api_key)

    def get_user_input(question_for_user: str):
        print("--------------------------QUESTION_FOR_USER--------------------------")
        print(question_for_user)
        print("---------------------------------------------------------------------")
        return input("Enter your input: ")

    async def run_main(question_for_user: str | None = None):
        if question_for_user:
            user_input = get_user_input(question_for_user)
        else:
            user_input = None

        user_input_needed = await main(model_config, user_input)

        if user_input_needed:
            await run_main(user_input_needed)
        else:
            # After all steps are complete, optionally open Google Maps
            print("Would you like me to open Google Maps for navigation? (y/n)")
            choice = input().strip().lower()
            if choice == "y":
                origin = model_config.get("last_known_location", "Current Location")
                destination = model_config.get("last_destination")
                if origin and destination:
                    map_url = f"https://www.google.com/maps/dir/?api=1&origin= {origin}&destination={destination}"
                    print(f"Opening Google Maps: {map_url}")
                    webbrowser.open(map_url)
                else:
                    print("Cannot open map: Missing origin or destination.")

    asyncio.run(run_main())