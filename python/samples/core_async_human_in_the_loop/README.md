# AutoGen Multi-Agent Personal Assistant

A sophisticated multi-agent personal assistant system built with AutoGen Core that **actually integrates** with specialized agents to provide real assistance with commute planning, project management, and calendar scheduling.

## 🚀 **Key Features**

- **✅ Real Agent Integration**: Actually calls and coordinates specialized agents (not dummy responses)
- **📡 Streaming Responses**: See AI responses stream in real-time character by character
- **👁️ Visible Agent Interactions**: Watch agent communications in the terminal with timestamps
- **🧠 Intelligent Routing**: Uses LLM to analyze requests and determine optimal agent coordination
- **💾 State Persistence**: Maintains conversation context across sessions
- **🔄 Human-in-the-Loop**: Interactive conversations with proper state management

## 🤖 **The Agent Team**

### 🎭 **Orchestrator**
Central coordinator that uses AI to analyze your requests and route them to the right specialist agents

### 🚗 **Commute Agent** 
- Real-time route planning and traffic analysis
- Google Maps integration for navigation
- Travel time calculations and alternative routes

### 📚 **Project Agent**
- Intelligent project breakdown into manageable tasks
- Timeline creation and milestone tracking
- Academic and work planning assistance

### 📅 **Calendar Agent**
- Meeting scheduling and conflict detection
- Google Calendar integration
- Event coordination and time management

## 📁 **Essential Files** (Cleaned Up)

```
core_async_human_in_the_loop/
├── orchestrator.py              # ✅ Main orchestrator with real agent integration
├── commute_agent.py             # 🚗 Commute/navigation specialist
├── project_agent.py             # 📚 Project management specialist  
├── calendar_agent.py            # 📅 Calendar/scheduling specialist
├── model_config.yml             # ⚙️ Your API configuration
├── model_config_template.yml    # 📋 Configuration template
├── requirements.txt             # 📦 Python dependencies
├── install.sh                   # 🔧 Automated setup script
├── run.sh                       # 🚀 Easy run script
├── README.md                    # 📖 This documentation
├── .gitignore                   # 🙈 Git ignore rules
├── user_data.json              # 👤 User preferences (created at runtime)
└── conversations.log           # 📜 Conversation history (created at runtime)
```
## 🚀 **Quick Start**

### Step 1: Setup
```bash
cd python/samples/core_async_human_in_the_loop

# Automated setup
./install.sh
```

### Step 2: Configure API Keys
Edit `model_config.yml` with your credentials:


**For Azure OpenAI and google api:**
```yaml
provider: autogen_ext.models.openai.AzureOpenAIChatCompletionClient
config:
  model: "gpt-4o"
  azure_endpoint: https://your-endpoint.openai.azure.com/
  azure_deployment: "your-deployment-name"
  api_version: "2024-02-01"
  api_key: YOUR_AZURE_API_KEY
google_maps_api_key: Your_Google_api_key

```

### Step 3: Run the System
```bash
./run.sh
```

## 💬 **What You'll See**

### Real-Time Agent Interactions
```
🤖 [14:23:15] → ORCHESTRATOR
   Analyzing request: Plan my commute to university tomorrow

🤖 [14:23:16] → COMMUTE AGENT  
   Processing: Plan my commute to university tomorrow...

🤖 [14:23:18] ← COMMUTE AGENT
   Route analysis complete - providing navigation recommendations

💬 [14:23:18] ORCHESTRATOR Response:
   🚗 **Commute Planning Results**
   
   **Route Information:**
   - From: 123 Main St, San Francisco, CA
   - To: Stanford University, CA
   - Mode: driving
   ...
```

## 🎯 **Usage Examples**

### Commute Planning
```
👤 Your response: I need to get to university by 9 AM tomorrow, what's the best route?

🤖 [14:25:10] → ORCHESTRATOR
   Analyzing request: I need to get to university by 9 AM tomorrow, what's the best route?

🤖 [14:25:11] → COMMUTE AGENT
   Processing: I need to get to university by 9 AM tomorrow...

💬 [14:25:13] ORCHESTRATOR Response:
   🚗 **Commute Planning Results**
   
   **Route Information:**
   - From: 123 Main St, San Francisco, CA
   - To: Stanford University, CA
   - Mode: driving
   - Departure Time: 8:15 AM (recommended)
   
   **Navigation Assistance:**
   Based on current traffic patterns, I recommend leaving at 8:15 AM to arrive by 9:00 AM...
```

### Project Management
```
👤 Your response: Break down my machine learning project due next Friday

🤖 [14:27:05] → PROJECT AGENT
   Processing: Break down my machine learning project due next Friday...

💬 [14:27:08] ORCHESTRATOR Response:
   📚 **Project Management Analysis**
   
   **Project Goal:** machine learning project
   **Deadline:** next Friday
   
   **Suggested Task Breakdown:**
   1. **Research and Planning** (Days 1-2)
      - Dataset selection and exploration
      - Algorithm research and comparison
   ...
```

### Calendar Scheduling
```
👤 Your response: Schedule a team meeting for 2 hours next week

🤖 [14:29:20] → CALENDAR AGENT
   Processing: Schedule a team meeting for 2 hours next week...

💬 [14:29:23] ORCHESTRATOR Response:
   📅 **Calendar Management Results**
   
   **Available Time Slots:**
   - Tuesday 10:00 AM - 12:00 PM
   - Wednesday 2:00 PM - 4:00 PM
   - Thursday 9:00 AM - 11:00 AM
   ...
```

## ⚙️ ** API Integrations**

### Google Maps API (Enhanced Commute Features)
1. Get API key from [Google Cloud Console](https://console.cloud.google.com/)
2. Enable Directions API and Distance Matrix API
3. Add to environment: `export GOOGLE_MAPS_API_KEY=your_key`

### Google Calendar API (Enhanced Scheduling)
1. Create OAuth credentials in [Google Cloud Console](https://console.cloud.google.com/)
2. Download `credentials.json` to project directory
3. First run will prompt for authentication

## 🛠️ **Manual Setup** (If Automated Fails)

```bash
# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Configure API keys
cp model_config_template.yml model_config.yml
# Edit model_config.yml with your API credentials

# Run the system
python3 orchestrator.py
```

## 🧠 **How It Works**

1. **User Input**: You provide a request in natural language
2. **AI Analysis**: The orchestrator uses LLM to analyze your request and extract context
3. **Agent Routing**: Requests are routed to appropriate specialist agents
4. **Real Processing**: Agents actually process your request and provide substantive responses
5. **Coordination**: Multiple agents can work together on complex requests
6. **Streaming Output**: Responses stream in real-time with visible agent interactions

## 🔧 **Troubleshooting**

### "ModuleNotFoundError: No module named 'autogen_core'"
```bash
source venv/bin/activate
pip install -r requirements.txt
```

### "model_config.yml not found"
```bash
cp model_config_template.yml model_config.yml
# Edit with your API credentials
```

### Agent Import Errors
Make sure all agent files are in the same directory and virtual environment is activated.

### Streaming Issues
If character-by-character streaming seems slow, you can adjust the delay in `print_streaming_response()` function.

## 📋 **Requirements**

- Python 3.9+
- OpenAI API key OR Azure OpenAI credentials
- Google Maps API key (optional, for enhanced commute features)
- Google Calendar API credentials (optional, for enhanced scheduling features)

## 🔄 **Architecture**

```
User Input
    ↓
🎭 Orchestrator (AI Analysis)
    ↓
📊 LLM Request Analysis
    ↓
🔀 Agent Routing Decision
    ↓
┌─────────────┬─────────────┬─────────────┐
│ 🚗 Commute  │ 📚 Project  │ 📅 Calendar │
│ Agent       │ Agent       │ Agent       │
└─────────────┴─────────────┴─────────────┘
    ↓
💬 Streaming Response with Agent Interactions
```

## 🎉 **What Makes This Special**

Unlike typical demo systems, this orchestrator:

- **Actually works**: Real agent integration, not placeholder responses
- **Shows its thinking**: Visible agent communications and reasoning
- **Streams naturally**: Real-time response streaming like ChatGPT
- **Learns context**: Remembers your preferences and previous conversations
- **Coordinates intelligently**: Uses AI to route requests optimally
- **Handles complexity**: Can coordinate multiple agents for complex requests

Ready to experience a truly intelligent multi-agent assistant? Run `./run.sh` and start exploring!
