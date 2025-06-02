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

@type_subscription("project_conversation")
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
            GetSlowUserMessage(content=message.content),
            topic_id=DefaultTopicId("project_conversation"),
        )

    async def save_state(self) -> Mapping[str, Any]:
        return {"memory": await self._model_context.save_state()}

    async def load_state(self, state: Mapping[str, Any]) -> None:
        await self._model_context.load_state(state["memory"])

class DecomposeProjectInput(BaseModel):
    goal: str = Field(description="Overall project goal")
    due_date: str = Field(description="Project due date")

class DecomposeProjectOutput(BaseModel):
    subtasks: list[Dict[str, str]] = Field(description="List of subtasks with their deadlines")

class DecomposeProjectTool(BaseTool[DecomposeProjectInput, DecomposeProjectOutput]):
    def __init__(self, model_client: ChatCompletionClient):
        super().__init__(
            DecomposeProjectInput,
            DecomposeProjectOutput,
            "decompose_project",
            "Decompose a project goal and due date into subtasks with timelines",
        )
        self._model_client = model_client

    async def run(self, args: DecomposeProjectInput, cancellation_token: CancellationToken) -> DecomposeProjectOutput:
        system = SystemMessage(content=(
            f"""You are a project management assistant. 
        Please break down the goal '{args.goal}' with the deadline '{args.due_date}' 
        into several subtasks, and provide the latest completion date for each task (in the format YYYY-MM-DD). 
        The output must be **only** a JSON array, where each element is of the form {{"task": "...", "deadline": "..."}}.
        Today's date is {datetime.datetime.now().strftime("%Y-%m-%d")}."""
        ))

        user = UserMessage(content="", source="user")
        response = await self._model_client.create(
            [system, user],
        )
  
        try:
            subtasks = json.loads(response.content)
        except json.JSONDecodeError:
            text = response.content
            start = text.find('[')
            end = text.rfind(']') + 1
            subtasks = json.loads(text[start:end])

        print(f"Decomposing project '{args.goal}' with due date {args.due_date}")
        return DecomposeProjectOutput(subtasks=subtasks)

@type_subscription("project_conversation")
class ProjectAgent(RoutedAgent):
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
I am a helpful AI assistant that decomposes project goals into subtasks with timelines.
If any information is missing, I will ask for it.

Today's date is {datetime.datetime.now().strftime("%Y-%m-%d")}
"""
            )
        ]


    @message_handler
    async def handle_message(self, message: UserTextMessage, ctx: MessageContext) -> None:
        await self._model_context.add_message(UserMessage(content=message.content, source=message.source))

        if message.content.strip().lower() in ["yes", "yes.", "confirmed"]:
            await self.publish_message(
                TerminateMessage(content="Subtasks confirmed. Project decomposition complete."),
                topic_id=DefaultTopicId("project_conversation")
            )
            return

        tools = [DecomposeProjectTool(self._model_client)]
        previous_messages = await self._model_context.get_messages()
        previous_subtask_response = next(
            (msg for msg in reversed(previous_messages)
            if isinstance(msg, AssistantMessage) and msg.content.strip().startswith("- ")),
            None
        )

        if previous_subtask_response:
            revision_request = message.content.strip()
            combined_prompt = f"""
    You previously generated the following subtasks:
    {previous_subtask_response.content}

    The user now requests:
    {revision_request}

    Please revise the task list accordingly, taking the original subtasks into account. Output only the updated JSON array.
    """
            input_args = DecomposeProjectInput(goal=combined_prompt, due_date="")
            tool = tools[0]
            output: DecomposeProjectOutput = await tool.run(input_args, ctx.cancellation_token)

            formatted = "If everything looks good, please reply 'yes' or 'confirmed'. If there are any issues, feel free to provide feedback.\n" + "\n".join(f"- {task['task']} (by {task['deadline']})" for task in output.subtasks)
            speech = AssistantTextMessage(content=formatted, source=self.metadata["type"])
            await self._model_context.add_message(AssistantMessage(content=formatted, source=self.metadata["type"]))
            await self.publish_message(speech, topic_id=DefaultTopicId("project_conversation"))
            return

        response = await self._model_client.create(
            self._system_messages + (await self._model_context.get_messages()),
            tools=tools,
        )

        if isinstance(response.content, list) and all(isinstance(item, FunctionCall) for item in response.content):
            for call in response.content:
                tool = next((t for t in tools if t.name == call.name), None)
                if tool is None:
                    raise ValueError(f"Tool not found: {call.name}")
                args = json.loads(call.arguments)

                output: DecomposeProjectOutput = await tool.run_json(args, ctx.cancellation_token)

                if output.subtasks:
                    formatted = "If everything looks good, please reply 'yes' or 'confirmed'. If there are any issues, feel free to provide feedback.\n" + "\n".join(f"- {task['task']} (by {task['deadline']})" for task in output.subtasks)
                else:
                    formatted = "No subtasks were generated."

                speech = AssistantTextMessage(content=formatted, source=self.metadata["type"])
                await self._model_context.add_message(AssistantMessage(content=formatted, source=self.metadata["type"]))
                await self.publish_message(speech, topic_id=DefaultTopicId("project_conversation"))

            return

        assert isinstance(response.content, str)
        speech = AssistantTextMessage(content=response.content, source=self.metadata["type"])
        await self._model_context.add_message(AssistantMessage(content=response.content, source=self.metadata["type"]))
        await self.publish_message(speech, topic_id=DefaultTopicId("project_conversation"))


    async def save_state(self) -> Mapping[str, Any]:
        return {"memory": await self._model_context.save_state()}

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
    global state_persister

    print("--------------------------------MAIN Project Agent--------------------------------")
    model_client = ChatCompletionClient.load_component(model_config)
    termination_handler = TerminationHandler()
    needs_user_input_handler = NeedsUserInputHandler()
    runtime = SingleThreadedAgentRuntime(intervention_handlers=[needs_user_input_handler, termination_handler])

    await SlowUserProxyAgent.register(runtime, "User", lambda: SlowUserProxyAgent("User", "I am a user"))
    initial_message = AssistantTextMessage(
        content="Please provide your project goal and due date, for example: 'I want to build a portfolio website by 2025-06-30'.",
        source="User",
    )
    await ProjectAgent.register(
        runtime,
        "ProjectAgent",
        lambda: ProjectAgent(
            "ProjectAgent",
            description="AI that decomposes project goals into subtasks",
            model_client=model_client,
            initial_message=initial_message,
        ),
    )

    if latest_user_input is not None:
        runtime_initiation_message = UserTextMessage(content=latest_user_input, source="User")
    else:
        runtime_initiation_message = initial_message

    state = state_persister.load_content()
    if state:
        await runtime.load_state(state)

    await runtime.publish_message(
        runtime_initiation_message,
        DefaultTopicId("project_conversation"),
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
    with open("model_config.yml") as f:
        model_config = yaml.safe_load(f)

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

    asyncio.run(run_main())
