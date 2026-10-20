# PYTHON — tool_use_demo.py
import os
from dotenv import load_dotenv

load_dotenv()
import anthropic, json, requests

client = anthropic.Anthropic()

# === Define tools ===
tools = [
    {
        "name": "calculate",
        "description": "Evaluate a mathematical expression. Use for any math.",
        "input_schema": {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "The math expression to evaluate, e.g. '2 + 2' or '500000 * 0.2'"
                }
            },
            "required": ["expression"]
        }
    },
    {
        "name": "get_weather",
        "description": "Get current weather for a city.",
        "input_schema": {
            "type": "object",
            "properties": {
                "city": {"type": "string", "description": "City name, e.g. 'Bangkok'"}
            },
            "required": ["city"]
        }
    },
    {
        "name": "control_lights",
        "description": "Control ALL Tapo L530 smart lights. The UI wrapper manages the device target directly.",
        "input_schema": {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "description": "The action to perform: 'discover' (list all lights on the wrapper), 'on', 'off', 'set_brightness', or 'set_color'",
                    "enum": ["discover", "on", "off", "set_brightness", "set_color"]
                },
                "level": {
                    "type": "integer",
                    "description": "Brightness level (1-100). Required if action is 'set_brightness'."
                },
                "color": {
                    "type": "string",
                    "description": "The color to set the light to (e.g. 'red', 'blue', 'warm white', 'daylight'). Required if action is 'set_color'."
                }
            },
            "required": ["action"]
        }
    }
]

# === Tool implementations ===
def execute_tool(name, inputs):
    if name == "calculate":
        try:
            result = eval(inputs["expression"])  # ⚠️ Use ast.literal_eval in production!
            return str(result)
        except Exception as e:
            return f"Error: {e}"
    elif name == "get_weather":
        city = inputs["city"]
        try:
            # 1. Geocode the city name to get coordinates
            geocode_url = f"https://geocoding-api.open-meteo.com/v1/search?name={city}&count=1&language=en&format=json"
            geo_response = requests.get(geocode_url)
            geo_response.raise_for_status()
            geo_data = geo_response.json()
            
            if not geo_data.get("results"):
                return f"Could not find coordinates for city: {city}"
                
            location = geo_data["results"][0]
            lat = location["latitude"]
            lon = location["longitude"]
            
            # 2. Get the weather using the coordinates
            weather_url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current=temperature_2m,relative_humidity_2m,weather_code&temperature_unit=celsius"
            weather_response = requests.get(weather_url)
            weather_response.raise_for_status()
            weather_data = weather_response.json()
            
            current = weather_data["current"]
            temp = current["temperature_2m"]
            humidity = current["relative_humidity_2m"]
            
            return f"Weather in {city}: {temp}°C, {humidity}% humidity."
        except Exception as e:
            return f"Error getting weather: {e}"
    elif name == "control_lights":
        action = inputs["action"]
        base_url = "http://localhost:8003/api/control"
        
        try:
            if action == "discover":
                resp = requests.post(base_url, json={"cmd": "status"})
                if resp.status_code == 200 and resp.json().get("ok"):
                    info = resp.json().get("result", {})
                    return f"Found connected Tapo light. Status: {info}"
                return f"Failed to get status via web wrapper: {resp.text}"
                
            elif action in ["on", "off"]:
                resp = requests.post(base_url, json={"cmd": action})
                if resp.status_code == 200 and resp.json().get("ok"):
                    return f"Successfully turned {action} the light."
                return f"Failed to turn {action}: {resp.text}"
                
            elif action == "set_brightness":
                level = inputs.get("level", 50)
                resp = requests.post(base_url, json={"cmd": "brightness", "value": level})
                if resp.status_code == 200 and resp.json().get("ok"):
                    return f"Successfully set brightness to {level}%."
                return f"Failed to set brightness: {resp.text}"
                
            elif action == "set_color":
                color = inputs.get("color", "").lower()
                # Determine if it's a color temperature instead of an RGB color
                if any(w in color for w in ["warm", "white", "daylight", "cool"]):
                    temp_map = {"warm": 2700, "soft": 3000, "white": 4000, "daylight": 5000, "cool": 6500}
                    temp = 4000
                    for k, v in temp_map.items():
                        if k in color: temp = v; break
                    resp = requests.post(base_url, json={"cmd": "color-temp", "value": temp})
                    if resp.status_code == 200 and resp.json().get("ok"):
                        return f"Successfully set color temperature to {temp}K."
                else:
                    # Basic HSV mapping for standard colors requested by the agent
                    hsv_map = {"red": 0, "orange": 30, "yellow": 60, "green": 120, "blue": 240, "purple": 270, "pink": 300}
                    hue = 0
                    for k, v in hsv_map.items():
                        if k in color: hue = v; break
                    resp = requests.post(base_url, json={"cmd": "color", "hue": hue, "saturation": 100, "value": 100})
                    if resp.status_code == 200 and resp.json().get("ok"):
                        return f"Successfully set color to {color}."
                return f"Failed to set color or color not recognized properly."
                
            return f"Unknown action: {action}"
            
        except requests.exceptions.ConnectionError:
            return f"Error: Cannot connect to the Flask wrapper at {base_url}. Ensure app.py is running on port 8003."
        except Exception as e:
            return f"Error controlling lights via wrapper: {e}"

# === Conversation loop with tool use ===
def ask(question):
    messages = [{"role": "user", "content": question}]
    
    while True:
        response = client.messages.create(
            model="claude-sonnet-4-5-20250929",
            max_tokens=1024,
            tools=tools,
            messages=messages
        )
        
        # Check if Claude wants to use tools
        tool_calls = [b for b in response.content if b.type == "tool_use"]
        
        if not tool_calls:
            # No tool calls — return text response
            return response.content[0].text
        
        # Execute each tool call
        messages.append({"role": "assistant", "content": response.content})
        
        for tool_call in tool_calls:
            print(f"  🔧 Using tool: {tool_call.name}({tool_call.input})")
            result = execute_tool(tool_call.name, tool_call.input)
            messages.append({
                "role": "user",
                "content": [{"type": "tool_result",
                             "tool_use_id": tool_call.id,
                             "content": result}]
            })

# Test it!
print(ask("turn off all lights"))

