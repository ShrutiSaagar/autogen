"""
Multi-Agent Personal Assistant Orchestrator using AutoGen Core
Real LLM integration with Azure OpenAI client routing
"""

import asyncio
import json
import os
import yaml
import time
from datetime import datetime
from typing import Any, Dict, Mapping, Optional
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

from autogen_core import (
    DefaultInterventionHandler,
    DefaultTopicId,
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

USER_DATA_FILE = "user_data.json"

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

def load_user_data():
    if os.path.exists(USER_DATA_FILE):
        with open(USER_DATA_FILE, 'r') as f:
            return json.load(f)
    return {}

def save_user_data(data):
    with open(USER_DATA_FILE, 'w') as f:
        json.dump(data, f, indent=2)

def print_agent_interaction(agent_name: str, message: str, is_request: bool = True):
    timestamp = datetime.now().strftime("%H:%M:%S")
    direction = "→" if is_request else "←"
    print(f"\n🤖 [{timestamp}] {direction} {agent_name.upper()}")
    print(f"   {message}")
    print("-" * 60)

def print_streaming_response(content: str, agent_name: str = "ORCHESTRATOR"):
    timestamp = datetime.now().strftime("%H:%M:%S")
    print(f"\n💬 [{timestamp}] {agent_name} Response:")
    print("   ", end="", flush=True)
    
    for char in content:
        print(char, end="", flush=True)
        time.sleep(0.002)
    print("\n" + "=" * 60)

class MockPersistence:
    def __init__(self):
        self._content: Mapping[str, Any] = {}

    def load_content(self) -> Mapping[str, Any]:
        return self._content

    def save_content(self, content: Mapping[str, Any]) -> None:
        self._content = content

state_persister = MockPersistence()

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

@type_subscription("orchestrator_conversation")
class UserProxyAgent(RoutedAgent):
    def __init__(self, name: str, description: str) -> None:
        super().__init__(description)
        self._model_context = BufferedChatCompletionContext(buffer_size=10)
        self._name = name

    @message_handler
    async def handle_message(self, message: AssistantTextMessage, ctx: MessageContext) -> None:
        await self._model_context.add_message(AssistantMessage(content=message.content, source=message.source))
        await self.publish_message(
            GetSlowUserMessage(content=message.content), 
            topic_id=DefaultTopicId("orchestrator_conversation")
        )

    async def save_state(self) -> Mapping[str, Any]:
        return {"memory": await self._model_context.save_state()}

    async def load_state(self, state: Mapping[str, Any]) -> None:
        await self._model_context.load_state(state["memory"])

class AgentManager:
    def __init__(self, model_client: ChatCompletionClient, user_data: Dict[str, Any]):
        self.model_client = model_client
        self.user_data = user_data
        
        self.commute_system_prompt = f"""You are a commute planning specialist. Help users with:
- Route optimization and travel time calculations
- Traffic analysis and alternative routes  
- Navigation assistance between locations
- Transport mode recommendations

User Context:
- Home: {user_data.get('home_address', 'Not provided')}
- Work: {user_data.get('work_address', 'Not provided')}
- University: {user_data.get('university_address', 'Not provided')}
- Preferred Transport: {user_data.get('preferred_commute_mode', 'driving')}

Provide practical, actionable commute advice. Today is {datetime.now().strftime('%Y-%m-%d')}."""

        self.project_system_prompt = f"""You are a project management specialist. Help users with:
- Breaking down complex projects into manageable tasks
- Creating realistic timelines and milestones
- Setting priorities and deadlines
- Academic and work project planning

Provide structured, actionable project management advice. Today is {datetime.now().strftime('%Y-%m-%d')}."""

        self.calendar_system_prompt = f"""You are a calendar and scheduling specialist. Help users with:
- Meeting coordination and scheduling
- Calendar conflict detection
- Event planning and time management
- Finding optimal time slots for activities

Provide practical scheduling solutions and time management advice. Today is {datetime.now().strftime('%Y-%m-%d')}."""

    async def route_to_agent(self, agent_type: str, query: str, context: Dict[str, Any]) -> str:
        print_agent_interaction(f"{agent_type.upper()} AGENT", f"Processing: {query[:50]}...")
        
        try:
            if agent_type == "commute":
                return await self._call_commute_agent(query, context)
            elif agent_type == "project":
                return await self._call_project_agent(query, context)
            elif agent_type == "calendar":
                return await self._call_calendar_agent(query, context)
            else:
                return f"Unknown agent type: {agent_type}"
        except Exception as e:
            error_msg = f"Error routing to {agent_type} agent: {str(e)}"
            print_agent_interaction(f"{agent_type.upper()} AGENT", f"❌ Error: {error_msg}", False)
            return error_msg

    async def _call_commute_agent(self, query: str, context: Dict[str, Any]) -> str:
        messages = [
            SystemMessage(content=self.commute_system_prompt),
            UserMessage(content=f"User request: {query}\nAdditional context: {json.dumps(context)}", source="user")
        ]
        
        response = await self.model_client.create(messages)
        print_agent_interaction("COMMUTE AGENT", "Generated route recommendations", False)
        return response.content

    async def _call_project_agent(self, query: str, context: Dict[str, Any]) -> str:
        messages = [
            SystemMessage(content=self.project_system_prompt),
            UserMessage(content=f"User request: {query}\nAdditional context: {json.dumps(context)}", source="user")
        ]
        
        response = await self.model_client.create(messages)
        print_agent_interaction("PROJECT AGENT", "Generated project breakdown", False)
        return response.content

    async def _call_calendar_agent(self, query: str, context: Dict[str, Any]) -> str:
        messages = [
            SystemMessage(content=self.calendar_system_prompt),
            UserMessage(content=f"User request: {query}\nAdditional context: {json.dumps(context)}", source="user")
        ]
        
        response = await self.model_client.create(messages)
        print_agent_interaction("CALENDAR AGENT", "Generated scheduling recommendations", False)
        return response.content

@type_subscription("orchestrator_conversation")
class OrchestratorAgent(RoutedAgent):
    def __init__(
        self, 
        name: str, 
        description: str, 
        model_client: ChatCompletionClient,
        agent_manager: AgentManager,
        initial_message: AssistantTextMessage | None = None
    ) -> None:
        super().__init__(description)
        self._model_context = BufferedChatCompletionContext(
            buffer_size=10,
            initial_messages=[UserMessage(content=initial_message.content, source=initial_message.source)]
            if initial_message
            else None,
        )
        self._name = name
        self._model_client = model_client
        self._agent_manager = agent_manager
        self._user_data = load_user_data()
        
        self._system_messages = [
            SystemMessage(
                content=f"""You are an intelligent orchestrator for a multi-agent personal assistant system. 

**Available Agents:**
1. **CommuteAgent**: Travel planning, route optimization, traffic analysis, navigation
2. **ProjectAgent**: Project breakdown, task scheduling, deadline management, planning
3. **CalendarAgent**: Meeting scheduling, calendar management, event planning

**User Context:**
- Home: {self._user_data.get('home_address', 'Not provided')}
- University: {self._user_data.get('university_address', 'Not provided')}
- Work: {self._user_data.get('work_address', 'Not provided')}
- Commute Preference: {self._user_data.get('preferred_commute_mode', 'Not provided')}

**Your Role:**
- Analyze user requests and determine which agent(s) to engage
- Extract relevant context for each agent
- Coordinate multiple agents when needed
- Provide unified responses combining agent outputs

**Decision Logic keyword help:**
- Keywords: 'travel', 'commute', 'directions', 'route', 'traffic' → CommuteAgent
- Keywords: 'project', 'assignment', 'deadline', 'task', 'goal', 'work' → ProjectAgent  
- Keywords: 'calendar', 'meeting', 'schedule', 'appointment', 'event' → CalendarAgent

Today is {datetime.now().strftime("%Y-%m-%d")}."""
            )
        ]

    @message_handler
    async def handle_message(self, message: UserTextMessage, ctx: MessageContext) -> None:
        await self._model_context.add_message(UserMessage(content=message.content, source=message.source))
        
        print_agent_interaction("ORCHESTRATOR", f"Analyzing request: {message.content}")
        
        analysis_prompt = f"""Analyze this user request and determine which agent(s) should handle it:

User Request: "{message.content}"

Available agents: commute, project, calendar

Respond with a JSON object:
{{
    "primary_agent": "agent_name",
    "secondary_agents": ["agent_name2"],
    "reasoning": "explanation",
    "extracted_context": {{
        "key": "value pairs"
    }}
}}

If unclear, use "none" for primary_agent."""

        try:
            analysis_response = await self._model_client.create([
                SystemMessage(content="You are a request analyzer. Always respond with valid JSON."),
                UserMessage(content=analysis_prompt, source="system")
            ])
            
            analysis = json.loads(analysis_response.content)
            primary_agent = analysis.get("primary_agent")
            secondary_agents = analysis.get("secondary_agents", [])
            extracted_context = analysis.get("extracted_context", {})
            reasoning = analysis.get("reasoning", "")
            
            print_agent_interaction("ORCHESTRATOR", f"Analysis: {reasoning}")
            
            response_parts = []
            
            if primary_agent and primary_agent != "none":
                primary_response = await self._agent_manager.route_to_agent(
                    primary_agent, message.content, extracted_context
                )
                response_parts.append(primary_response)
                
                for secondary_agent in secondary_agents:
                    if secondary_agent != primary_agent:
                        secondary_response = await self._agent_manager.route_to_agent(
                            secondary_agent, message.content, extracted_context
                        )
                        response_parts.append(f"\n**Additional {secondary_agent.title()} Information:**\n{secondary_response}")
            
            if not response_parts:
                response_content = await self._generate_general_response(message.content)
            else:
                response_content = "\n\n".join(response_parts)
                
        except Exception as e:
            print_agent_interaction("ORCHESTRATOR", f"❌ Error in request analysis: {str(e)}", False)
            response_content = await self._generate_fallback_response(message.content)
        
        if any(keyword in message.content.lower() for keyword in ['exit', 'quit', 'goodbye', 'bye']):
            await self.publish_message(
                TerminateMessage(content="Thank you for using the Multi-Agent Assistant! Goodbye!"),
                topic_id=DefaultTopicId("orchestrator_conversation")
            )
            return
        
        print_streaming_response(response_content, "ORCHESTRATOR")
        
        speech = AssistantTextMessage(content=response_content, source=self.metadata["type"])
        await self._model_context.add_message(AssistantMessage(content=response_content, source=self.metadata["type"]))
        await self.publish_message(speech, topic_id=DefaultTopicId("orchestrator_conversation"))

    async def _generate_general_response(self, user_input: str) -> str:
        messages = self._system_messages + [
            UserMessage(content=f"User is asking: '{user_input}'. Provide helpful guidance about what you can assist with.", source="user")
        ]
        
        response = await self._model_client.create(messages)
        return response.content

    async def _generate_fallback_response(self, user_input: str) -> str:
        messages = [
            SystemMessage(content="You are a helpful assistant. The user's request couldn't be analyzed properly, so provide general guidance."),
            UserMessage(content=f"User request: '{user_input}'. Explain what you can help with.", source="user")
        ]
        
        response = await self._model_client.create(messages)
        return response.content

    async def save_state(self) -> Mapping[str, Any]:
        return {
            "memory": await self._model_context.save_state(),
            "user_data": self._user_data
        }

    async def load_state(self, state: Mapping[str, Any]) -> None:
        await self._model_context.load_state(state["memory"])
        self._user_data = state.get("user_data", {})

async def run_orchestrator(model_config: Dict[str, Any], latest_user_input: Optional[str] = None) -> Optional[str]:
    global state_persister

    print_agent_interaction("SYSTEM", "Loading AI model client...")
    model_client = ChatCompletionClient.load_component(model_config)

    termination_handler = TerminationHandler()
    needs_user_input_handler = NeedsUserInputHandler()
    runtime = SingleThreadedAgentRuntime(intervention_handlers=[needs_user_input_handler, termination_handler])

    user_data = load_user_data()
    agent_manager = AgentManager(model_client, user_data)

    await UserProxyAgent.register(runtime, "User", lambda: UserProxyAgent("User", "Human user proxy"))

    initial_message = AssistantTextMessage(
        content="Welcome to your Multi-Agent Personal Assistant! I can help you with commute planning, project management, and calendar scheduling. How can I assist you today?",
        source="System"
    )
    
    await OrchestratorAgent.register(
        runtime,
        "Orchestrator",
        lambda: OrchestratorAgent(
            "Orchestrator",
            description="Multi-agent orchestrator for personal assistance",
            model_client=model_client,
            agent_manager=agent_manager,
            initial_message=initial_message if latest_user_input is None else None,
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
        DefaultTopicId("orchestrator_conversation"),
    )

    runtime.start()
    await runtime.stop_when(lambda: termination_handler.is_terminated or needs_user_input_handler.needs_user_input)
    await model_client.close()

    user_input_needed = None
    if needs_user_input_handler.user_input_content is not None:
        user_input_needed = needs_user_input_handler.user_input_content
    elif termination_handler.is_terminated:
        print_streaming_response(f"Session ended: {termination_handler.termination_msg}", "SYSTEM")

    state_to_persist = await runtime.save_state()
    state_persister.save_content(state_to_persist)

    return user_input_needed

async def ainput(prompt: str = "") -> str:
    with ThreadPoolExecutor(1, "AsyncInput") as executor:
        return await asyncio.get_event_loop().run_in_executor(executor, input, prompt)

def main():
    print("🤖 " + "="*50)
    print("   AUTOGEN MULTI-AGENT PERSONAL ASSISTANT")
    print("="*52)
    print("   Real LLM Integration | Azure OpenAI Client")
    print("="*52 + "\n")
    
    user_data = load_user_data()

    if not all(key in user_data for key in ['home_address', 'university_address', 'work_address', 'preferred_commute_mode']):
        print("🔧 INITIAL SETUP")
        print("Please provide some basic information (you'll only need to do this once):\n")
        
        if 'home_address' not in user_data or not user_data['home_address']:
            user_data['home_address'] = input("Enter your home address: ")
        if 'university_address' not in user_data or not user_data['university_address']:
            user_data['university_address'] = input("Enter your university address: ")
        if 'work_address' not in user_data or not user_data['work_address']:
            user_data['work_address'] = input("Enter your work address: ")
        if 'preferred_commute_mode' not in user_data or not user_data['preferred_commute_mode']:
            user_data['preferred_commute_mode'] = input("Enter preferred commute mode (e.g., driving, public_transit): ")
        
        save_user_data(user_data)
        print()

    try:
        with open("model_config.yml") as f:
            model_config = yaml.safe_load(f)
    except FileNotFoundError:
        print("❌ Error: model_config.yml not found. Please create it using model_config_template.yml as a template.")
        return

    def get_user_input(question_for_user: str):
        print_streaming_response(question_for_user, "ASSISTANT")
        user_input = input("\n👤 Your response: ")
        return user_input

    async def run_main_loop(question_for_user: str | None = None):
        if question_for_user:
            user_input = get_user_input(question_for_user)
        else:
            user_input = None
            
        try:
            user_input_needed = await run_orchestrator(model_config, user_input)
            if user_input_needed:
                await run_main_loop(user_input_needed)
        except KeyboardInterrupt:
            print_streaming_response("Session interrupted by user. Goodbye!", "SYSTEM")
            return
        except Exception as e:
            print_streaming_response(f"Error occurred: {e}\nPlease try again or contact support.", "ERROR")

    print("🚀 SYSTEM READY")
    print("="*60)
    print("Available capabilities:")
    print("🚗 Commute planning and navigation")
    print("📚 Project management and task breakdown") 
    print("📅 Calendar scheduling and event management")
    print("\nType 'exit' or 'quit' to end the session.")
    print("="*60)

    try:
        asyncio.run(run_main_loop())
    except KeyboardInterrupt:
        print_streaming_response("Session ended by user. Thank you for using the Multi-Agent Assistant!", "SYSTEM")

if __name__ == "__main__":
    main()