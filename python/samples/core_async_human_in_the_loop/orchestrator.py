"""
Multi-Agent Personal Assistant Orchestrator using AutoGen Core
Real LLM integration with Azure OpenAI client routing
"""

import asyncio
import json
import os
import yaml
import time
import logging
from datetime import datetime
from typing import Any, Dict, Mapping, Optional
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

# Import session logger
from session_logger import get_session_logger, start_new_session, end_current_session

# Configure comprehensive logging for agent interactions
# Set VERBOSE_LOGGING to False to reduce console output
VERBOSE_LOGGING = True

# Configure file handler for detailed logging
file_handler = logging.FileHandler('agent_interactions.log')
file_handler.setLevel(logging.INFO)
file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(file_formatter)

# Configure console handler with reduced verbosity
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.WARNING if not VERBOSE_LOGGING else logging.INFO)
console_formatter = logging.Formatter('%(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(console_formatter)

# Configure root logger
logging.basicConfig(
    level=logging.INFO,
    handlers=[file_handler, console_handler]
)

# Suppress verbose autogen_core logs
logging.getLogger('autogen_core').setLevel(logging.WARNING)
logging.getLogger('autogen_core.models').setLevel(logging.WARNING)
logging.getLogger('autogen_core._routed_agent').setLevel(logging.WARNING)
logging.getLogger('autogen_core._base_agent').setLevel(logging.WARNING)
logging.getLogger('autogen_core._agent_runtime').setLevel(logging.WARNING)
logging.getLogger('autogen_core.model_context').setLevel(logging.WARNING)

# Keep our custom agent interaction logger at INFO level
agent_logger = logging.getLogger('AgentInteractions')
agent_logger.setLevel(logging.INFO)

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

from calendar_agent import run_calendar_agent
from project_agent import main as run_project_agent
from commute_agent import main as run_commute_agent

USER_DATA_FILE = "user_data.json"

with open("model_config.yml") as f:
    model_config = yaml.safe_load(f)

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
    console_msg = f"\n🤖 [{timestamp}] {direction} {agent_name.upper()}"
    console_detail = f"   {message}"
    
    print(console_msg)
    print(console_detail)
    print("-" * 60)
    
    # Enhanced logging for debugging
    log_msg = f"{direction} {agent_name.upper()}: {message}"
    agent_logger.info(log_msg)

def log_agent_response_detailed(agent_name: str, request: str, response: str, context: Dict[str, Any] = None):
    """Log detailed agent interactions for debugging purposes."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    log_entry = {
        "timestamp": timestamp,
        "agent": agent_name,
        "request": request,
        "response": response,
        "context_keys": list(context.keys()) if context else [],
        "response_length": len(response)
    }
    
    # Log to file for detailed analysis
    agent_logger.info(f"DETAILED_INTERACTION: {json.dumps(log_entry, indent=2)}")
    
    # Only show full response in console if verbose logging is enabled
    if VERBOSE_LOGGING:
        print(f"\n📋 FULL {agent_name.upper()} RESPONSE:")
        print("="*80)
        print(response)
        print("="*80)
    else:
        # Just show a summary in console
        print(f"\n📋 {agent_name.upper()} Response: {len(response)} chars (logged to file)")

def log_orchestrator_decision(iteration: int, analysis: Dict[str, Any], reasoning: str):
    """Log orchestrator decision-making process."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    log_entry = {
        "timestamp": timestamp,
        "iteration": iteration,
        "primary_agent": analysis.get("primary_agent"),
        "secondary_agents": analysis.get("secondary_agents", []),
        "reasoning": reasoning,
        "extracted_context_keys": list(analysis.get("extracted_context", {}).keys())
    }
    
    agent_logger.info(f"ORCHESTRATOR_DECISION: {json.dumps(log_entry, indent=2)}")
    
    # Simplified console output
    print(f"\n🧠 ORCHESTRATOR DECISION (Iteration {iteration}):")
    print(f"   Primary Agent: {analysis.get('primary_agent')}")
    print(f"   Reasoning: {reasoning[:100]}..." if len(reasoning) > 100 else f"   Reasoning: {reasoning}")
    if analysis.get("secondary_agents"):
        print(f"   Secondary Agents: {analysis.get('secondary_agents')}")
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
        
        self.commute_system_prompt = f"""🚗 **You are a commute planning specialist.** Help users with:
- 🗺️ **Route optimization** and travel time calculations
- 🚦 **Traffic analysis** and alternative routes  
- 🧭 **Navigation assistance** between locations
- 🚌 **Transport mode recommendations**

## 👤 **User Context:**
- 🏠 **Home:** {user_data.get('home_address', 'Not provided')}
- 🏢 **Work:** {user_data.get('work_address', 'Not provided')}
- 🎓 **University:** {user_data.get('university_address', 'Not provided')}
- 🚗 **Preferred Transport:** {user_data.get('preferred_commute_mode', 'driving')}

## 📝 **Response Guidelines:**
Always respond using **markdown formatting** with:
- 📊 **Clear headers and sections**
- 🗺️ **Map links** and navigation URLs when possible
- ⏱️ **Time estimates** with traffic considerations
- 🚦 **Traffic alerts** and road conditions
- 🛣️ **Alternative routes** with pros/cons
- 📍 **Step-by-step directions** when helpful
- 🎯 **Emojis** for visual clarity

**Important:** When providing directions, always include **clickable map links** (Google Maps, Apple Maps, etc.) so users can easily navigate. Format links as: 🗺️ **[View on Google Maps](URL)**

Provide practical, actionable commute advice. 📅 Today is {datetime.now().strftime('%Y-%m-%d')}."""

        self.project_system_prompt = f"""📋 **You are a project management specialist.** Help users with:
- 🎯 **Breaking down complex projects** into manageable tasks
- ⏰ **Creating realistic timelines** and milestones
- 📊 **Setting priorities** and deadlines
- 🎓 **Academic and work project planning**

## 📝 **Response Guidelines:**
Always respond using **markdown formatting** with:
- 📊 **Clear headers and sections**
- ✅ **Task breakdowns** with checkboxes
- 📅 **Timeline visuals** and milestone markers
- 🎯 **Priority indicators** (High/Medium/Low)
- 📈 **Progress tracking** suggestions
- ⚠️ **Risk assessments** and mitigation strategies
- 🎯 **Emojis** for visual clarity

Provide structured, actionable project management advice. 📅 Today is {datetime.now().strftime('%Y-%m-%d')}."""

        self.calendar_system_prompt = f"""📅 **You are a calendar and scheduling specialist.** Help users with:
- 🤝 **Meeting coordination** and scheduling
- ⚠️ **Calendar conflict detection**
- 🎉 **Event planning** and time management
- ⏰ **Finding optimal time slots** for activities
- 🔄 **Creating multiple events** in batch operations for efficiency

## 🔄 **Multiple Events Handling:**
When handling requests for multiple events:
- 🛠️ **Use the create_multiple_events tool** for batch operations
- ✅ **Ensure all event details are complete** before processing
- 📊 **Provide comprehensive summaries** of batch operations
- 🤝 **Handle both successful and failed** event creations gracefully

## 📝 **Response Guidelines:**
Always respond using **markdown formatting** with:
- 📊 **Clear headers and sections**
- 📅 **Event details** in organized blocks
- 🔗 **Clickable calendar links**
- ✅ **Success/failure indicators**
- ⏰ **Time conflicts** and availability info
- 🎯 **Emojis** for visual clarity

Provide practical scheduling solutions and time management advice. 📅 Today is {datetime.now().strftime('%Y-%m-%d')}."""

        self.refine_query_system_prompt = f"""You are a prompt refinement specialist. Help users with:
- Refining user requests to be more specific and clear in one sentence
- This should entail context from the previous user chat history and the current user request
- This will be used to refine the user request and to send to the agent
- If the calendar query dates are not specified, you should assume the user is asking for the next 7 days


Provide the final refined user request. Today is {datetime.now().strftime('%Y-%m-%d')}."""

        self.completion_evaluator_prompt = f"""You are a completion evaluator. Your job is to determine if a user's request has been fully satisfied based on the conversation history and agent responses.

Analyze the original user request and all agent responses to determine completion status.

Consider a request COMPLETE if:
- All requested information has been provided
- All requested actions have been performed successfully
- The user's question has been answered comprehensively
- No follow-up actions are obviously needed

Consider a request INCOMPLETE if:
- Key information is missing
- Actions failed and need retry
- The response is vague or unhelpful
- Additional context or clarification is needed

Respond with a JSON object:
{{
    "is_complete": true/false,
    "reasoning": "detailed explanation of why complete or incomplete",
    "next_action_needed": "specific suggestion for next step if incomplete",
    "confidence": 0.8
}}

Today is {datetime.now().strftime('%Y-%m-%d')}."""

        self.smart_request_analyzer_prompt = f"""You are an intelligent request analyzer that can identify ambiguous calendar requests and determine what information needs to be gathered.

Analyze user requests for calendar operations and identify:
1. What the user wants to do (schedule, view, edit, batch_schedule, etc.)
2. What information is missing or ambiguous
3. What assumptions can be made
4. What information needs to be gathered from other sources
5. Whether this is a request for multiple events/batch processing

For ambiguous scheduling requests, consider:
- Default meeting duration (30-60 minutes for business, 15-30 for quick calls)
- Working hours (9 AM - 6 PM weekdays)
- Avoid conflicts with existing events
- Use placeholder recipients if not specified

For multiple events requests:
- Detect keywords like "multiple", "several", "batch", "recurring", "series"
- Identify if user wants to create more than one event
- Determine if events are related (same type, recurring pattern)
- Check if sufficient details are provided for all events

Respond with a JSON object:
{{
    "request_type": "schedule|view|edit|batch_schedule|other",
    "is_multiple_events": true/false,
    "event_count": 1,
    "is_ambiguous": true/false,
    "missing_info": ["date", "time", "recipient", "duration", "event_id"],
    "assumptions_to_make": {{
        "duration": 30,
        "time_preference": "morning|afternoon|any",
        "default_title": "Meeting"
    }},
    "info_gathering_needed": {{
        "get_events": "to find available slots or get event IDs",
        "get_calendar_context": "to avoid conflicts"
    }},
    "suggested_defaults": {{
        "date": "next_business_day",
        "time": "10:00",
        "duration": 30,
        "title": "Team Meeting"
    }},
    "batch_processing_plan": {{
        "use_batch_creation": true/false,
        "requires_pattern_detection": true/false,
        "suggested_time_spacing": "1_hour",
        "batch_description": "Weekly team meetings"
    }}
}}

Today is {datetime.now().strftime('%Y-%m-%d')}."""

    async def route_to_agent(self, agent_type: str, query: str, context: Dict[str, Any]) -> str:
        print_agent_interaction(f"{agent_type.upper()} AGENT", f"Processing: {query[:50]}...")
        
        # Get session logger
        session_logger = get_session_logger()
        
        try:
            # Log the original request
            agent_logger.info(f"ROUTING_TO_{agent_type.upper()}: Original query: {query}")
            
            query = await self.refine_query(query, context)
            
            # Log the refined query
            agent_logger.info(f"ROUTING_TO_{agent_type.upper()}: Refined query: {query}")
            
            if agent_type == "commute":
                response = await self._call_commute_agent(query, context)
            elif agent_type == "project":
                response = await self._call_project_agent(query, context)
            elif agent_type == "calendar":
                response = await self._call_calendar_agent(query, context)
            else:
                error_msg = f"Unknown agent type: {agent_type}"
                agent_logger.error(f"ROUTING_ERROR: {error_msg}")
                session_logger.log_error("ROUTING_ERROR", error_msg, {"agent_type": agent_type, "query": query})
                return error_msg
            
            # Log detailed response
            log_agent_response_detailed(agent_type, query, response, context)
            
            # Log to session logger
            session_logger.log_agent_interaction(agent_type, query, response, is_success=True)
            
            return response
            
        except Exception as e:
            error_msg = f"Error routing to {agent_type} agent: {str(e)}"
            agent_logger.error(f"ROUTING_EXCEPTION_{agent_type.upper()}: {error_msg}")
            print_agent_interaction(f"{agent_type.upper()} AGENT", f" Error: {error_msg}", False)
            
            # Log error to session logger
            session_logger.log_error(f"AGENT_ROUTING_ERROR", error_msg, {
                "agent_type": agent_type, 
                "query": query,
                "context_keys": list(context.keys())
            })
            
            return error_msg

    async def _call_commute_agent(self, query: str, context: Dict[str, Any]) -> str:
        if VERBOSE_LOGGING:
            print("Calling commute agent, query: ", query, "context: ", context)
        
        # Enhance the query with conversation context since project agent doesn't accept context parameter
        conversation_context = context.get("full_conversation", [])
        if conversation_context:
            recent_context = conversation_context[-3:]  # Last 3 exchanges for context
            context_summary = "\n".join(recent_context)
            enhanced_query = f"Given recent conversation context:\n{context_summary}\n\nCurrent request: {query}"
        else:
            enhanced_query = query
            
        response = await run_commute_agent(model_config, enhanced_query)
        print_agent_interaction("COMMUTE AGENT", "Generated route recommendations", False)
        
        if hasattr(response, 'content'):
            return response.content
        else:
            return str(response)

    async def _call_project_agent(self, query: str, context: Dict[str, Any]) -> str:
        if VERBOSE_LOGGING:
            print("Calling project agent, query: ", query, "context: ", context)
        
        # Enhance the query with conversation context since project agent doesn't accept context parameter
        conversation_context = context.get("full_conversation", [])
        if conversation_context:
            recent_context = conversation_context[-3:]  # Last 3 exchanges for context
            context_summary = "\n".join(recent_context)
            enhanced_query = f"Given recent conversation context:\n{context_summary}\n\nCurrent request: {query}"
        else:
            enhanced_query = query
        
        response = await run_project_agent(model_config, enhanced_query)
        print_agent_interaction("PROJECT AGENT", "Generated project breakdown", False)
        
        # Handle both string responses and response objects
        if hasattr(response, 'content'):
            return response.content
        else:
            return str(response)

    async def _call_calendar_agent(self, query: str, context: Dict[str, Any]) -> str:
        if VERBOSE_LOGGING:
            print("Calling calendar agent, query: ", query, "context: ", context)
        
        # Enhance the query with conversation context since calendar agent doesn't accept context parameter
        conversation_context = context.get("full_conversation", [])
        if conversation_context:
            recent_context = conversation_context[-3:]  # Last 3 exchanges for context
            context_summary = "\n".join(recent_context)
            enhanced_query = f"Given recent conversation context:\n{context_summary}\n\nCurrent request: {query}"
        else:
            enhanced_query = query
        
        response = await run_calendar_agent(model_config, enhanced_query)
        if VERBOSE_LOGGING:
            print("Calendar agent response: ", response)
        
        # Handle both string responses and response objects
        if hasattr(response, 'content'):
            return response.content
        else:
            return str(response)
    
    async def refine_query(self, query: str, context: Dict[str, Any]) -> str:
        # Include conversation context in query refinement
        conversation_context = context.get("full_conversation", [])
        context_summary = ""
        if conversation_context:
            recent_context = conversation_context[-5:]  # Last 5 exchanges for context
            context_summary = f"\n\nRecent conversation context:\n{chr(10).join(recent_context)}"
        
        enhanced_prompt = self.refine_query_system_prompt + context_summary
        
        messages = [
            SystemMessage(content=enhanced_prompt),
            UserMessage(content=f"User request: {query}\nAdditional context: {json.dumps({k: v for k, v in context.items() if k != 'full_conversation'})}", source="user")
        ]
        
        response = await self.model_client.create(messages)
        
        # Handle both string responses and response objects  
        if hasattr(response, 'content'):
            return response.content
        else:
            return str(response)

    async def evaluate_completion(self, original_request: str, conversation_history: list, agent_responses: list) -> Dict[str, Any]:
        """Evaluate if the user request has been fully satisfied."""
        conversation_summary = "\n".join([f"- {msg}" for msg in conversation_history[-5:]])  # Last 5 messages
        responses_summary = "\n".join([f"Agent Response {i+1}: {resp}" for i, resp in enumerate(agent_responses)])
        
        evaluation_content = f"""
Original User Request: "{original_request}"

Recent Conversation History:
{conversation_summary}

Agent Responses:
{responses_summary}

Evaluate if the original request has been fully satisfied.
        """
        
        messages = [
            SystemMessage(content=self.completion_evaluator_prompt),
            UserMessage(content=evaluation_content, source="system")
        ]
        
        try:
            response = await self.model_client.create(messages)
            evaluation = json.loads(response.content)
            print_agent_interaction("COMPLETION EVALUATOR", f"Request completion: {evaluation.get('is_complete')} - {evaluation.get('reasoning')}")
            
            # Log to session logger
            session_logger = get_session_logger()
            session_logger.log_completion_evaluation(evaluation, original_request)
            
            return evaluation
        except Exception as e:
            print_agent_interaction("COMPLETION EVALUATOR", f" Error in evaluation: {str(e)}", False)
            
            # Log error to session logger
            session_logger = get_session_logger()
            session_logger.log_error("COMPLETION_EVALUATION_ERROR", str(e), {
                "original_request": original_request,
                "conversation_history_length": len(conversation_history),
                "agent_responses_count": len(agent_responses)
            })
            
            # Default to complete if evaluation fails to prevent infinite loops
            return {
                "is_complete": True,
                "reasoning": "Evaluation failed, defaulting to complete",
                "next_action_needed": "",
                "confidence": 0.5
            }

    async def analyze_smart_request(self, user_request: str, conversation_history: list = None) -> Dict[str, Any]:
        """Analyze user request to identify ambiguities and required information gathering."""
        history_context = ""
        if conversation_history:
            recent_history = conversation_history[-3:]  # Last 3 messages for context
            history_context = f"\nRecent conversation context:\n{chr(10).join(recent_history)}"
        
        analysis_content = f"""
User Request: "{user_request}"
{history_context}

Analyze this request for calendar operations, identify ambiguities, and determine what information needs to be gathered.
        """
        
        messages = [
            SystemMessage(content=self.smart_request_analyzer_prompt),
            UserMessage(content=analysis_content, source="system")
        ]
        
        try:
            response = await self.model_client.create(messages)
            analysis = json.loads(response.content)
            print_agent_interaction("SMART ANALYZER", f"Request type: {analysis.get('request_type')} | Ambiguous: {analysis.get('is_ambiguous')}")
            return analysis
        except Exception as e:
            print_agent_interaction("SMART ANALYZER", f" Error in smart analysis: {str(e)}", False)
            # Return basic analysis as fallback
            return {
                "request_type": "other",
                "is_ambiguous": False,
                "missing_info": [],
                "assumptions_to_make": {},
                "info_gathering_needed": {},
                "suggested_defaults": {}
            }

    async def gather_missing_info(self, missing_info: list, suggested_defaults: dict, conversation_history: list) -> Dict[str, Any]:
        """Gather missing information by calling appropriate agents."""
        gathered_info = {}
        
        # If we need calendar context or event information
        if any(info in missing_info for info in ["event_id", "available_time", "conflicts"]):
            try:
                # Get recent events to understand context
                from datetime import datetime, timedelta
                today = datetime.now()
                start_date = today.strftime("%Y-%m-%d")
                end_date = (today + timedelta(days=7)).strftime("%Y-%m-%d")
                
                print_agent_interaction("INFO GATHERER", "Fetching calendar context for smart scheduling")
                calendar_context = await self._call_calendar_agent(
                    f"Show me events from {start_date} to {end_date}",
                    {"purpose": "gathering_context", "for_smart_scheduling": True}
                )
                
                gathered_info["calendar_context"] = calendar_context
                gathered_info["recent_events"] = calendar_context
                
                # Extract available time slots (simplified logic)
                if "No events found" in calendar_context:
                    gathered_info["available_slots"] = ["09:00", "10:00", "11:00", "14:00", "15:00"]
                else:
                    # Basic conflict detection - could be enhanced
                    gathered_info["available_slots"] = ["10:00", "11:00", "14:00", "15:00"]
                    
            except Exception as e:
                print_agent_interaction("INFO GATHERER", f"Error gathering calendar context: {str(e)}", False)
                gathered_info["available_slots"] = ["10:00", "14:00"]
        
        # Apply suggested defaults for missing information
        for info_key in missing_info:
            if info_key in suggested_defaults and info_key not in gathered_info:
                gathered_info[info_key] = suggested_defaults[info_key]
        
        return gathered_info

    async def create_smart_calendar_request(self, original_request: str, smart_analysis: dict, gathered_info: dict) -> str:
        """Create an enhanced calendar request with all necessary information."""
        request_type = smart_analysis.get("request_type", "schedule")
        missing_info = smart_analysis.get("missing_info", [])
        defaults = smart_analysis.get("suggested_defaults", {})
        is_multiple_events = smart_analysis.get("is_multiple_events", False)
        event_count = smart_analysis.get("event_count", 1)
        batch_processing_plan = smart_analysis.get("batch_processing_plan", {})
        
        if request_type == "batch_schedule" or is_multiple_events:
            # Handle multiple events creation
            enhanced_request = f"Create multiple events with the following details:\n\n"
            
            # Generate events based on analysis
            for i in range(event_count):
                enhanced_request += f"Event {i+1}:\n"
                
                # Add title/summary
                title = gathered_info.get("title", defaults.get("title", f"Meeting {i+1}"))
                enhanced_request += f"- Summary: {title}\n"
                
                # Add date (spread events if multiple)
                base_date = gathered_info.get("date", defaults.get("date", "tomorrow"))
                if base_date == "next_business_day":
                    from datetime import datetime, timedelta
                    today = datetime.now()
                    next_business = today + timedelta(days=1 + i)  # Spread across days
                    while next_business.weekday() >= 5:  # Skip weekends
                        next_business += timedelta(days=1)
                    date = next_business.strftime("%Y-%m-%d")
                else:
                    date = base_date
                enhanced_request += f"- Date: {date}\n"
                
                # Add time (space them out if same day)
                available_slots = gathered_info.get("available_slots", ["10:00", "11:00", "14:00", "15:00"])
                time_index = i % len(available_slots)
                time = gathered_info.get("time", available_slots[time_index])
                enhanced_request += f"- Time: {time}\n"
                
                # Add duration
                duration = gathered_info.get("duration", defaults.get("duration", 30))
                enhanced_request += f"- Duration: {duration} minutes\n"
                
                # Add recipient
                recipient = gathered_info.get("recipient", f"TBD - Recipient {i+1}")
                enhanced_request += f"- Recipient: {recipient}\n"
                
                # Add location
                location = gathered_info.get("location", "TBD - Location")
                enhanced_request += f"- Location: {location}\n"
                
                # Add description
                description = gathered_info.get("description", f"Scheduled meeting {i+1}")
                enhanced_request += f"- Description: {description}\n\n"
            
            # Add batch description
            batch_description = batch_processing_plan.get("batch_description", f"Batch creation of {event_count} events")
            enhanced_request += f"Batch Description: {batch_description}\n\n"
            enhanced_request += "NOTE: These are placeholder events that will need recipient and location confirmation."
            
        elif request_type == "schedule":
            # Build a comprehensive scheduling request
            enhanced_request = f"Schedule a meeting"
            
            # Add title/summary
            title = gathered_info.get("title", defaults.get("title", "Team Meeting"))
            enhanced_request += f" titled '{title}'"
            
            # Add date
            date = gathered_info.get("date", defaults.get("date", "tomorrow"))
            if date == "next_business_day":
                from datetime import datetime, timedelta
                today = datetime.now()
                next_business = today + timedelta(days=1)
                while next_business.weekday() >= 5:  # Weekend
                    next_business += timedelta(days=1)
                date = next_business.strftime("%Y-%m-%d")
            enhanced_request += f" on {date}"
            
            # Add time (prefer available slots)
            available_slots = gathered_info.get("available_slots", ["10:00"])
            time = gathered_info.get("time", available_slots[0] if available_slots else "10:00")
            enhanced_request += f" at {time}"
            
            # Add duration
            duration = gathered_info.get("duration", defaults.get("duration", 30))
            enhanced_request += f" for {duration} minutes"
            
            # Add recipient (placeholder if not provided)
            recipient = gathered_info.get("recipient", "TBD - Recipient")
            enhanced_request += f" with {recipient}"
            
            # Add location if specified
            location = gathered_info.get("location", "TBD - Location")
            enhanced_request += f" at {location}"
            
            # Add note about placeholder values
            enhanced_request += ". NOTE: This creates a placeholder event that will need recipient confirmation."
            
        elif request_type == "edit":
            # For edit requests, we might need to find the event first
            if "event_id" in missing_info and gathered_info.get("calendar_context"):
                enhanced_request = f"First show me recent events to identify which event to edit, then: {original_request}"
            else:
                enhanced_request = original_request
                
        elif request_type == "view":
            # Enhance view requests with reasonable date ranges
            if not any(word in original_request.lower() for word in ["today", "tomorrow", "week", "month", "from", "to"]):
                enhanced_request = f"{original_request} for the next 7 days"
            else:
                enhanced_request = original_request
        else:
            enhanced_request = original_request
            
        return enhanced_request

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
                content=f"""🤖 **You are an intelligent orchestrator** for a multi-agent personal assistant system that facilitates collaborative problem-solving between specialized agents and users.

## 🎯 **Available Agents:**
1. 🚗 **CommuteAgent**: Travel planning, route optimization, traffic analysis, navigation
2. 📋 **ProjectAgent**: Project breakdown, task scheduling, deadline management, planning
3. 📅 **CalendarAgent**: Meeting scheduling, calendar management, event planning, multiple events creation

## 👤 **User Context:**
- 🏠 **Home:** {self._user_data.get('home_address', 'Not provided')}
- 🎓 **University:** {self._user_data.get('university_address', 'Not provided')}
- 🏢 **Work:** {self._user_data.get('work_address', 'Not provided')}
- 🚗 **Commute Preference:** {self._user_data.get('preferred_commute_mode', 'Not provided')}

## 🛠️ **Your Enhanced Role & Capabilities:**
- 🎯 **Multi-Agent Coordinator**: Analyze user requests and determine which agent(s) to engage
- 📚 **Context Manager**: Extract and maintain relevant context for each agent interaction
- ❓ **Information Gatherer**: Ask users for additional information when needed to complete tasks
- 🤝 **Agent Liaison**: Pass specific requests from agents back to users when clarification is needed
- 🔄 **Iterative Problem Solver**: Work with agents across multiple iterations to accomplish complex goals
- 💬 **Conversation Facilitator**: Enable smooth information flow between user, agents, and yourself
- 📦 **Batch Processing Coordinator**: When users request multiple events or similar batch operations, coordinate with agents to process them efficiently

## 🔄 **Enhanced Decision-Making Process:**
1. 🔍 **Analyze** the user's complete request in context of the full conversation
2. 🎯 **Identify** what information is needed and what's missing
3. 🤝 **Coordinate** with appropriate agents to gather information and solutions
4. ❓ **Request** additional details from users when agents need clarification
5. 🔄 **Iterate** between agents and users until the goal is accomplished
6. 📝 **Synthesize** all information into a comprehensive response

## 🔄 **Multiple Events & Batch Processing:**
When users request multiple meetings, events, or similar batch operations:
- 🔍 **Detect Batch Requests**: Identify requests like "schedule 3 meetings", "create events for the week", "set up recurring meetings"
- 📊 **Gather Complete Information**: Collect all necessary details for each event in the batch
- 📤 **Structured Communication**: Send complete, structured data to the CalendarAgent in the following format:
  ```
  Create multiple events with the following details:
  Event 1: [complete details including date, time, duration, recipient, summary, location]
  Event 2: [complete details]
  Event 3: [complete details]
  Batch Description: [purpose/context of these events]
  ```
- ✅ **Comprehensive Processing**: Ensure all events have complete information before sending to avoid back-and-forth
- 📊 **Results Coordination**: Handle batch results including successes and failures appropriately

## 👤 **User Interaction Guidelines:**
- ❓ Ask clarifying questions when user requests are ambiguous
- ℹ️ Request specific information that agents need to complete tasks
- 💡 Explain why additional information is needed
- 🎯 Offer multiple options when appropriate
- ✅ Confirm important decisions with users before proceeding
- 📦 For batch operations, gather all details upfront to minimize iterations

## 🤝 **Agent Coordination Guidelines:**
- 🔄 Use agents iteratively - one agent's output can inform another agent's input
- 📤 Pass agent questions/requirements back to users
- 🏗️ Coordinate multi-step workflows across different agents
- 📊 Gather all necessary information before final execution
- 🧠 Always consider the full conversation context when making decisions
- 📦 For multiple events, send complete structured data in one request to CalendarAgent
- 🗺️ **Important for Commute Agent**: Always ensure map links and navigation URLs are passed through to users

## 🎯 **Example Workflows:**
- 📋📅 **Project + Calendar**: Use ProjectAgent to break down tasks, then CalendarAgent to schedule them
- 🚗📅 **Commute + Calendar**: Check calendar for appointments, then plan optimal commute routes
- 🤝 **Multi-agent consultation**: Get input from multiple agents before presenting final recommendation
- 📦📅 **Batch Calendar Operations**: Gather all event details, then create multiple events in one CalendarAgent call

## 🔍 **Decision Logic Keywords:**
- 🚗 'travel', 'commute', 'directions', 'route', 'traffic' → CommuteAgent
- 📋 'project', 'assignment', 'deadline', 'task', 'goal', 'work' → ProjectAgent  
- 📅 'calendar', 'meeting', 'schedule', 'appointment', 'event' → CalendarAgent
- 📦 'multiple meetings', 'batch schedule', 'several events', 'recurring meetings' → CalendarAgent with batch processing

## 📝 **Response Guidelines:**
Always respond using **markdown formatting** with:
- 📊 **Clear headers and sections**
- 🎯 **Emojis** for visual clarity and engagement
- 🔗 **Proper link formatting** for any URLs from agents
- ✅ **Success/failure indicators**
- 📈 **Organized information** presentation

**🚨 Important**: You can and should ask users for additional information, pass agent requests to users, and coordinate multiple iterations to ensure complete task accomplishment. Don't hesitate to request clarification or additional details. For batch operations, prioritize gathering complete information upfront.

📅 **Today is {datetime.now().strftime("%Y-%m-%d")}.**"""
            )
        ]

    @message_handler
    async def handle_message(self, message: UserTextMessage, ctx: MessageContext) -> None:
        await self._model_context.add_message(UserMessage(content=message.content, source=message.source))
        original_request = message.content
        
        # Get session logger and log user message
        session_logger = get_session_logger()
        session_logger.log_user_message(original_request)
        
        print_agent_interaction("ORCHESTRATOR", f"Starting smart multi-iteration processing for: {original_request}")
        
        # Get full conversation history from model context for continuity
        full_conversation_context = await self._get_conversation_context()
        
        # Initialize tracking variables
        iteration_count = 0
        max_iterations = 5  # Safety limit to prevent infinite loops
        conversation_history = full_conversation_context  # Use existing context instead of resetting
        agent_responses = []
        all_response_parts = []
        user_confirmation_needed = False
        
        # First, perform smart analysis to identify if this is an ambiguous calendar request
        # Pass full conversation context for better analysis
        smart_analysis = await self._agent_manager.analyze_smart_request(original_request, conversation_history)
        
        # If it's an ambiguous calendar request, gather missing information proactively
        if smart_analysis.get("is_ambiguous", False) and smart_analysis.get("request_type") in ["schedule", "edit"]:
            print_agent_interaction("ORCHESTRATOR", "🧠 Detected ambiguous request - gathering missing information")
            
            missing_info = smart_analysis.get("missing_info", [])
            suggested_defaults = smart_analysis.get("suggested_defaults", {})
            
            # Gather missing information from other agents/sources
            gathered_info = await self._agent_manager.gather_missing_info(missing_info, suggested_defaults, conversation_history)
            
            # Create an enhanced request with all gathered information
            enhanced_request = await self._agent_manager.create_smart_calendar_request(
                original_request, smart_analysis, gathered_info
            )
            
            print_agent_interaction("ORCHESTRATOR", f"📝 Enhanced request: {enhanced_request[:100]}...")
            
            # Update the working request for processing
            working_request = enhanced_request
            user_confirmation_needed = True  # Always need confirmation for enhanced requests
        else:
            working_request = original_request
        
        while iteration_count < max_iterations:
            iteration_count += 1
            print_agent_interaction("ORCHESTRATOR", f"=== ITERATION {iteration_count} ===")
            
            # Analyze current state and determine next action with full context
            analysis_prompt = f"""Analyze the current state and determine the next action needed for collaborative problem-solving:

Original User Request: "{original_request}"
Working Request: "{working_request}"
Iteration: {iteration_count}

FULL Conversation History (for context):
{chr(10).join(conversation_history)}

Previous Agent Responses (this session):
{chr(10).join([f"Response {i+1}: {resp[:100]}..." for i, resp in enumerate(agent_responses)])}

Available agents: commute, project, calendar

**Enhanced Analysis Framework:**
1. **Completeness Check**: Is the user request fully understood? What information is missing?
2. **Agent Requirements**: What do the agents need to provide a complete solution?
3. **User Information Needs**: What additional details should we ask the user?
4. **Multi-Agent Coordination**: Can multiple agents work together on this request?
5. **Iteration Planning**: What's the optimal next step in the problem-solving process?

**Decision Options:**
- **Agent Action**: Route to specific agent(s) for information/action
- **User Query**: Ask user for additional information or clarification
- **Multi-Agent**: Coordinate between multiple agents
- **Completion**: Task is complete and ready for final response
- **Clarification**: Need to clarify requirements with user before proceeding

Consider the FULL conversation context and prioritize user experience.
If agents need specific information that only the user can provide, request it.
If multiple agents could contribute, consider using them collaboratively.

Respond with a JSON object:
{{
    "action_type": "agent_action|user_query|multi_agent|completion|clarification",
    "primary_agent": "agent_name or null",
    "secondary_agents": ["agent_name2"],
    "user_question": "specific question to ask user if action_type is user_query",
    "reasoning": "detailed explanation of why this action is needed",
    "extracted_context": {{
        "key": "value pairs including any relevant info from conversation and previous responses"
    }},
    "query_focus": "specific aspect to focus on this iteration",
    "missing_info": ["list of information still needed"],
    "coordination_plan": "how agents/user will work together"
}}

If no further action is needed, use "completion" for action_type."""

            try:
                # Get analysis for this iteration with full context
                analysis_response = await self._model_client.create([
                    SystemMessage(content="You are a request analyzer for a collaborative multi-agent system. Always respond with valid JSON. Consider the full conversation context and enable user interaction when needed."),
                    UserMessage(content=analysis_prompt, source="system")
                ])
                
                analysis = json.loads(analysis_response.content)
                action_type = analysis.get("action_type", "agent_action")
                primary_agent = analysis.get("primary_agent")
                secondary_agents = analysis.get("secondary_agents", [])
                user_question = analysis.get("user_question")
                extracted_context = analysis.get("extracted_context", {})
                extracted_context["full_conversation"] = conversation_history  # Include full context
                reasoning = analysis.get("reasoning", "")
                query_focus = analysis.get("query_focus", working_request)
                missing_info = analysis.get("missing_info", [])
                coordination_plan = analysis.get("coordination_plan", "")
                
                # Log orchestrator decision
                log_orchestrator_decision(iteration_count, analysis, reasoning)
                
                # Log to session logger
                session_logger.log_orchestrator_analysis(iteration_count, analysis, reasoning)
                
                # Handle different action types
                if action_type == "completion":
                    print_agent_interaction("ORCHESTRATOR", "✅ Analysis indicates task is complete")
                    break
                
                elif action_type == "user_query" or action_type == "clarification":
                    if user_question:
                        print_agent_interaction("ORCHESTRATOR", f"👤 Requesting user input: {user_question}")
                        # Publish a message requesting user input
                        await self.publish_message(
                            GetSlowUserMessage(content=f"I need some additional information to help you better:\n\n{user_question}\n\nPlease provide the requested details so I can coordinate with the appropriate agents to accomplish your goal."),
                            topic_id=DefaultTopicId("orchestrator_conversation")
                        )
                        # Add to conversation history and continue to next iteration
                        conversation_history.append(f"Orchestrator requested: {user_question}")
                        continue
                    else:
                        print_agent_interaction("ORCHESTRATOR", "⚠️ User query needed but no question specified")
                        break
                
                elif action_type == "agent_action" or action_type == "multi_agent":
                    # Execute agent calls for this iteration
                    iteration_responses = []
                    
                    if primary_agent:
                        try:
                            primary_response = await self._agent_manager.route_to_agent(
                                primary_agent, query_focus, extracted_context
                            )
                            iteration_responses.append(primary_response)
                            conversation_history.append(f"Orchestrator → {primary_agent.title()}: {query_focus}")
                            conversation_history.append(f"{primary_agent.title()} Response: {primary_response[:100]}...")
                            
                        except Exception as e:
                            error_msg = f"Error in {primary_agent} agent: {str(e)}"
                            iteration_responses.append(error_msg)
                            conversation_history.append(f"Error: {error_msg}")
                            print_agent_interaction("ORCHESTRATOR", f" {error_msg}", False)
                    
                    # Execute secondary agents if specified
                    for secondary_agent in secondary_agents:
                        if secondary_agent != primary_agent:
                            try:
                                secondary_response = await self._agent_manager.route_to_agent(
                                    secondary_agent, query_focus, extracted_context
                                )
                                secondary_formatted = f"\n**Additional {secondary_agent.title()} Information:**\n{secondary_response}"
                                iteration_responses.append(secondary_formatted)
                                conversation_history.append(f"Orchestrator → {secondary_agent.title()}: {query_focus}")
                                conversation_history.append(f"{secondary_agent.title()} Response: {secondary_response[:100]}...")
                                
                            except Exception as e:
                                error_msg = f"Error in {secondary_agent} agent: {str(e)}"
                                iteration_responses.append(f"\n**{secondary_agent.title()} Error:** {error_msg}")
                                conversation_history.append(f"Error: {error_msg}")
                                print_agent_interaction("ORCHESTRATOR", f" {error_msg}", False)
                    
                    # Store responses from this iteration
                    if iteration_responses:
                        combined_response = "\n\n".join(iteration_responses)
                        agent_responses.append(combined_response)
                        all_response_parts.extend(iteration_responses)
                        
                        # Log coordination plan if provided
                        if coordination_plan:
                            agent_logger.info(f"COORDINATION_PLAN_{iteration_count}: {coordination_plan}")
                    
                    # After agent execution, evaluate if the request is now complete
                    if action_type in ["agent_action", "multi_agent"] and agent_responses:
                        evaluation = await self._agent_manager.evaluate_completion(
                            original_request, conversation_history, agent_responses
                        )
                        
                        # If request is complete or confidence is high, break
                        if evaluation.get("is_complete", False) or evaluation.get("confidence", 0) > 0.9:
                            print_agent_interaction("ORCHESTRATOR", f"✅ Request completed after {iteration_count} iterations")
                            break
                        elif iteration_count >= max_iterations - 1:
                            print_agent_interaction("ORCHESTRATOR", f"⚠️ Reached max iterations ({max_iterations}), completing")
                            break
                        else:
                            next_action = evaluation.get("next_action_needed", "")
                            print_agent_interaction("ORCHESTRATOR", f"Continuing to iteration {iteration_count + 1}: {next_action}")
                            
                            # Add the evaluation result to conversation history for next iteration
                            conversation_history.append(f"Evaluation: {evaluation.get('reasoning', '')}")
                            if next_action:
                                conversation_history.append(f"Next needed: {next_action}")
                    
                    # Check iteration limit for non-agent actions too
                    elif iteration_count >= max_iterations - 1:
                        print_agent_interaction("ORCHESTRATOR", f"⚠️ Reached max iterations ({max_iterations}), completing")
                        break
                
                else:
                    print_agent_interaction("ORCHESTRATOR", f"⚠️ Unknown action type: {action_type}")
                    break
            
            except Exception as e:
                error_msg = f"Error in iteration {iteration_count}: {str(e)}"
                print_agent_interaction("ORCHESTRATOR", f" {error_msg}", False)
                agent_responses.append(error_msg)
                all_response_parts.append(error_msg)
                break
        
        # Handle exit conditions
        if any(keyword in original_request.lower() for keyword in ['exit', 'quit', 'goodbye', 'bye']):
            await self.publish_message(
                TerminateMessage(content="Thank you for using the Multi-Agent Assistant! Goodbye!"),
                topic_id=DefaultTopicId("orchestrator_conversation")
            )
            return
        
        # Prepare final response
        if not all_response_parts:
            response_content = await self._generate_general_response(original_request, conversation_history)
        else:
            # Combine all responses with iteration markers
            if len(agent_responses) == 1:
                # Single iteration - clean response
                response_content = "\n\n".join(all_response_parts)
            else:
                # Multiple iterations - show progression
                response_content = f"**Completed after {iteration_count} iterations:**\n\n" + "\n\n".join(all_response_parts)
        
        # Add user confirmation request if needed
        if user_confirmation_needed and smart_analysis.get("request_type") == "schedule":
            confirmation_request = f"""

**⚠️ CONFIRMATION NEEDED:**
I've created a placeholder meeting based on your request. Please confirm or modify:

📅 **Meeting Details Created:**
- **When:** Please check the details above
- **Duration:** Please verify the time
- **Recipients:** Please specify who should attend
- **Location:** Please confirm if location is correct

Would you like to:
1. Confirm the meeting as-is
2. Change the time/date
3. Add/modify recipients
4. Update location or other details
5. Cancel the meeting

Please let me know what adjustments you'd like to make!"""
            
            response_content += confirmation_request
        
        print_streaming_response(response_content, "ORCHESTRATOR")
        
        speech = AssistantTextMessage(content=response_content, source=self.metadata["type"])
        await self._model_context.add_message(AssistantMessage(content=response_content, source=self.metadata["type"]))
        await self.publish_message(speech, topic_id=DefaultTopicId("orchestrator_conversation"))

    async def _get_conversation_context(self) -> list:
        """Extract conversation history from model context for continuity."""
        try:
            # Get the messages from the model context
            messages = self._model_context._messages
            context_history = []
            
            for msg in messages:
                if isinstance(msg, UserMessage):
                    context_history.append(f"User: {msg.content}")
                elif isinstance(msg, AssistantMessage):
                    context_history.append(f"Assistant: {msg.content}")
                elif isinstance(msg, SystemMessage):
                    # Skip system messages in conversation history to avoid clutter
                    continue
            
            return context_history
        except Exception as e:
            print_agent_interaction("ORCHESTRATOR", f"Warning: Could not extract conversation context: {str(e)}", False)
            return []

    async def _generate_general_response(self, user_input: str, conversation_history: list = None) -> str:
        messages = self._system_messages.copy()
        
        # Include conversation context if available
        if conversation_history:
            context_summary = "\n".join(conversation_history[-5:])  # Last 5 exchanges
            context_message = UserMessage(
                content=f"Recent conversation context:\n{context_summary}\n\nCurrent user request: '{user_input}'. Provide helpful guidance considering the conversation context.",
                source="user"
            )
        else:
            context_message = UserMessage(
                content=f"User is asking: '{user_input}'. Provide helpful guidance about what you can assist with.",
                source="user"
            )
        
        messages.append(context_message)
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
    
    # Start new session logging
    session_logger = start_new_session()
    session_info = session_logger.get_session_info()
    print(f"📝 Session logging started: {session_info['log_file']}")
    print(f"📊 Session ID: {session_info['session_id']}\n")
    
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
        print(" Error: model_config.yml not found. Please create it using model_config_template.yml as a template.")
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
            end_current_session("Session completed successfully.")
        except KeyboardInterrupt:
            print_streaming_response("Session ended by user. Thank you for using the Multi-Agent Assistant!", "SYSTEM")
            end_current_session("Session ended by user.")

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
        end_current_session("Session ended by user.")

if __name__ == "__main__":
    main()