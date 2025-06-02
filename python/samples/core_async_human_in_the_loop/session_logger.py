"""
Session-based Markdown Logging System for Multi-Agent Assistant
Creates human-readable markdown log files for each session
"""

import os
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional


class SessionLogger:
    def __init__(self, logs_dir: str = "session_logs"):
        self.logs_dir = Path(logs_dir)
        self.logs_dir.mkdir(exist_ok=True)
        
        # Generate session ID based on timestamp
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_start = datetime.now()
        
        # Create session-specific log file
        self.log_file = self.logs_dir / f"session_{self.session_id}.md"
        
        # Initialize the session
        self._initialize_session_log()
    
    def _initialize_session_log(self):
        """Initialize the markdown log file with session header."""
        header = f"""# 🤖 Multi-Agent Assistant Session Log

## 📋 Session Information
- **Session ID:** `{self.session_id}`
- **Start Time:** {self.session_start.strftime("%Y-%m-%d %H:%M:%S")}
- **Log File:** `{self.log_file.name}`

---

## 💬 Conversation History

"""
        with open(self.log_file, 'w', encoding='utf-8') as f:
            f.write(header)
    
    def log_user_message(self, message: str, timestamp: Optional[datetime] = None):
        """Log a user message."""
        if timestamp is None:
            timestamp = datetime.now()
        
        log_entry = f"""### 👤 User Message
**Time:** {timestamp.strftime("%H:%M:%S")}

{message}

---

"""
        self._append_to_log(log_entry)
    
    def log_orchestrator_analysis(self, iteration: int, analysis: Dict[str, Any], reasoning: str):
        """Log orchestrator decision-making process."""
        timestamp = datetime.now()
        
        primary_agent = analysis.get("primary_agent", "Unknown")
        secondary_agents = analysis.get("secondary_agents", [])
        action_type = analysis.get("action_type", "Unknown")
        
        log_entry = f"""### 🧠 Orchestrator Analysis (Iteration {iteration})
**Time:** {timestamp.strftime("%H:%M:%S")}

#### 🎯 Decision Details
- **Action Type:** `{action_type}`
- **Primary Agent:** 🤖 **{primary_agent.title() if primary_agent else 'None'}**
- **Secondary Agents:** {', '.join([f'🤖 **{agent.title()}**' for agent in secondary_agents]) if secondary_agents else 'None'}

#### 💭 Reasoning
{reasoning}

#### 📊 Analysis Data
```json
{json.dumps(analysis, indent=2)}
```

---

"""
        self._append_to_log(log_entry)
    
    def log_agent_interaction(self, agent_name: str, request: str, response: str, is_success: bool = True):
        """Log agent interaction with request and response."""
        timestamp = datetime.now()
        
        status_emoji = "✅" if is_success else "❌"
        status_text = "Success" if is_success else "Error"
        
        # Determine agent emoji based on type
        agent_emoji = {
            "commute": "🚗",
            "project": "📋", 
            "calendar": "📅",
            "orchestrator": "🤖"
        }.get(agent_name.lower(), "🤖")
        
        log_entry = f"""### {agent_emoji} {agent_name.title()} Agent {status_emoji}
**Time:** {timestamp.strftime("%H:%M:%S")} | **Status:** {status_text}

#### 📤 Request
{request}

#### 📥 Response
{response}

---

"""
        self._append_to_log(log_entry)
    
    def log_error(self, error_type: str, error_message: str, context: Optional[Dict[str, Any]] = None):
        """Log errors with context."""
        timestamp = datetime.now()
        
        log_entry = f"""### ❌ Error Occurred
**Time:** {timestamp.strftime("%H:%M:%S")} | **Type:** {error_type}

#### 🚨 Error Message
```
{error_message}
```
"""
        
        if context:
            log_entry += f"""
#### 📋 Context
```json
{json.dumps(context, indent=2)}
```
"""
        
        log_entry += "\n---\n\n"
        self._append_to_log(log_entry)
    
    def log_completion_evaluation(self, evaluation: Dict[str, Any], original_request: str):
        """Log completion evaluation results."""
        timestamp = datetime.now()
        
        is_complete = evaluation.get("is_complete", False)
        confidence = evaluation.get("confidence", 0.0)
        reasoning = evaluation.get("reasoning", "No reasoning provided")
        next_action = evaluation.get("next_action_needed", "")
        
        status_emoji = "✅" if is_complete else "🔄"
        status_text = "Complete" if is_complete else "Incomplete"
        
        log_entry = f"""### 📊 Completion Evaluation {status_emoji}
**Time:** {timestamp.strftime("%H:%M:%S")} | **Status:** {status_text} | **Confidence:** {confidence:.1%}

#### 📝 Original Request
{original_request}

#### 💭 Evaluation Reasoning
{reasoning}
"""
        
        if next_action:
            log_entry += f"""
#### 🎯 Next Action Needed
{next_action}
"""
        
        log_entry += "\n---\n\n"
        self._append_to_log(log_entry)
    
    def log_batch_operation(self, operation_type: str, details: Dict[str, Any]):
        """Log batch operations like multiple events creation."""
        timestamp = datetime.now()
        
        log_entry = f"""### 📦 Batch Operation: {operation_type}
**Time:** {timestamp.strftime("%H:%M:%S")}

#### 📊 Operation Details
```json
{json.dumps(details, indent=2)}
```

---

"""
        self._append_to_log(log_entry)
    
    def log_session_end(self, summary: Optional[str] = None):
        """Log session end with optional summary."""
        end_time = datetime.now()
        duration = end_time - self.session_start
        
        log_entry = f"""

---

## 🏁 Session Summary

### 📊 Session Statistics
- **End Time:** {end_time.strftime("%Y-%m-%d %H:%M:%S")}
- **Duration:** {duration}
- **Total Interactions:** See conversation history above

### 📝 Session Summary
{summary if summary else "Session completed successfully."}

---

*Session log generated by Multi-Agent Assistant System*
"""
        self._append_to_log(log_entry)
    
    def _append_to_log(self, content: str):
        """Append content to the log file."""
        try:
            with open(self.log_file, 'a', encoding='utf-8') as f:
                f.write(content)
        except Exception as e:
            print(f"Failed to write to log file: {e}")
    
    def get_session_info(self) -> Dict[str, Any]:
        """Get current session information."""
        return {
            "session_id": self.session_id,
            "start_time": self.session_start.isoformat(),
            "log_file": str(self.log_file),
            "logs_directory": str(self.logs_dir)
        }


class MarkdownFormatter:
    """Utility class for formatting content in markdown."""
    
    @staticmethod
    def format_code_block(content: str, language: str = "") -> str:
        """Format content as a code block."""
        return f"```{language}\n{content}\n```"
    
    @staticmethod
    def format_link(text: str, url: str) -> str:
        """Format a markdown link."""
        return f"[{text}]({url})"
    
    @staticmethod
    def format_table(headers: list, rows: list) -> str:
        """Format a markdown table."""
        table = "| " + " | ".join(headers) + " |\n"
        table += "| " + " | ".join(["---"] * len(headers)) + " |\n"
        
        for row in rows:
            table += "| " + " | ".join(str(cell) for cell in row) + " |\n"
        
        return table
    
    @staticmethod
    def format_list(items: list, ordered: bool = False) -> str:
        """Format a markdown list."""
        if ordered:
            return "\n".join(f"{i+1}. {item}" for i, item in enumerate(items))
        else:
            return "\n".join(f"- {item}" for item in items)
    
    @staticmethod
    def format_alert(message: str, alert_type: str = "info") -> str:
        """Format an alert/callout."""
        emoji_map = {
            "info": "ℹ️",
            "warning": "⚠️", 
            "error": "❌",
            "success": "✅",
            "tip": "💡"
        }
        
        emoji = emoji_map.get(alert_type, "ℹ️")
        return f"{emoji} **{alert_type.title()}:** {message}"


# Global session logger instance
_session_logger: Optional[SessionLogger] = None

def get_session_logger() -> SessionLogger:
    """Get the current session logger instance."""
    global _session_logger
    if _session_logger is None:
        _session_logger = SessionLogger()
    return _session_logger

def start_new_session() -> SessionLogger:
    """Start a new logging session."""
    global _session_logger
    if _session_logger is not None:
        _session_logger.log_session_end("Session ended to start new session.")
    
    _session_logger = SessionLogger()
    return _session_logger

def end_current_session(summary: Optional[str] = None):
    """End the current logging session."""
    global _session_logger
    if _session_logger is not None:
        _session_logger.log_session_end(summary)
        _session_logger = None 