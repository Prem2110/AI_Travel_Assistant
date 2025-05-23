import os
import json
import requests
from datetime import datetime, timedelta
from typing import TypedDict, List, Dict, Any, Optional, Annotated
import re
from functools import lru_cache

# LangChain & LangGraph imports
from langchain.tools import BaseTool
from pydantic import BaseModel, Field
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.graph import StateGraph, END
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from dotenv import load_dotenv

load_dotenv(override=True)

# Set RapidAPI key from environment or use default
RAPIDAPI_KEY = os.getenv("RAPIDAPI_KEY", "01bc210b26msh2eb4ca8a4bb4f2bp130b38jsn89030bf3e39a")

# Initialize caches
location_cache = {}  # Cache for location IDs

# Use the OpenAI chat model (fallback if gen_ai_hub isn't available)
try:
    from gen_ai_hub.proxy.langchain.openai import ChatOpenAI
    LLM_DEPLOYMENT_ID = "d38dd2015862a15d"
    llm = ChatOpenAI(deployment_id=LLM_DEPLOYMENT_ID)
except ImportError:
    from langchain_openai import ChatOpenAI
    llm = ChatOpenAI(model_name="gpt-3.5-turbo")

# === STATE DEFINITION ===
class TravelState(TypedDict):
    """State type for the travel booking agent."""
    messages: List[Dict[str, Any]]
    travel_details: Dict[str, Any]
    flight_data: Optional[Dict[str, Any]]
    hotel_data: Optional[Dict[str, Any]]
    itinerary: Optional[Dict[str, Any]]
    error: Optional[str]
    next: Optional[str]

# === TOOLS ===
def get_city_location_id(city_name, api_type="flight"):
    """Get location ID with caching to avoid redundant API calls.

    Args:
        city_name: Name of the city (string or dict with 'name' key)
        api_type: Type of API to use ('flight' or 'hotel')
    """
    # Handle dictionary input by converting to string representation
    if isinstance(city_name, dict):
        # Extract city name from dictionary structure
        if "name" in city_name:
            city_str = city_name["name"]
        else:
            # Create a string from the first value or the whole dict
            city_str = next(iter(city_name.values())) if city_name else "unknown"
    else:
        city_str = str(city_name)

    # Check cache first
    cache_key = f"{city_str}_{api_type}"
    if cache_key in location_cache:
        print(f"📦 Using cached location ID for {city_str}")
        return location_cache[cache_key]

    # Different endpoints for flights and hotels
    if api_type == "flight":
        url = "https://booking-com15.p.rapidapi.com/api/v1/flights/searchDestination"
    else:  # hotel
        url = "https://booking-com15.p.rapidapi.com/api/v1/hotels/searchDestination"

    headers = {
        "X-RapidAPI-Key": RAPIDAPI_KEY,
        "X-RapidAPI-Host": "booking-com15.p.rapidapi.com"
    }

    try:
        print(f"🔍 Searching for {api_type} location ID for {city_str}...")
        # For API call, use city string - not the full dictionary
        params = {"query": city_str}
        res = requests.get(url, headers=headers, params=params, timeout=30)
        res.raise_for_status()
        data = res.json()

        if data.get('data') and len(data['data']) > 0:
            location_id = data['data'][0]['id']
            # Store in cache
            location_cache[cache_key] = location_id
            return location_id
        return None
    except Exception as e:
        print(f"📍 Error fetching location ID for {city_str}: {str(e)}")
        return None

def get_hotel_location_id(city_name):
    """Get hotel location ID with multiple fallback methods and caching."""
    # Handle city name extraction from string or dict
    city_str = ""
    if isinstance(city_name, dict):
        if "name" in city_name:
            city_str = city_name["name"]
        else:
            city_str = next(iter(city_name.values())) if city_name else "unknown"
    else:
        city_str = str(city_name)
    
    # Try first method (searchDestination endpoint)
    dest_id = get_city_location_id(city_str, api_type="hotel")

    # If found, return it
    if dest_id:
        return dest_id

    # Try with just the city name if the full location format was provided
    if city_str and ',' in city_str:
        simple_city = city_str.split(',')[0].strip()
        print(f"Trying with simplified city name: {simple_city}")
        dest_id = get_city_location_id(simple_city, api_type="hotel")
        if dest_id:
            return dest_id

    # Try alternative locations API as last resort
    url = "https://booking-com15.p.rapidapi.com/api/v1/hotels/locations"
    headers = {
        "X-RapidAPI-Key": RAPIDAPI_KEY,
        "X-RapidAPI-Host": "booking-com15.p.rapidapi.com"
    }

    # Get the simplest city name possible for the API
    simple_name = city_str.split(',')[0].strip()

    params = {"name": simple_name, "locale": "en-us"}
    try:
        print(f"Trying locations API for {simple_name}...")
        res = requests.get(url, headers=headers, params=params, timeout=30)
        res.raise_for_status()
        data = res.json()

        if data and isinstance(data, list) and len(data) > 0:
            for item in data:
                if item.get("dest_type") == "city":
                    location_id = item.get("dest_id")
                    # Store in cache
                    cache_key = f"{city_str}_hotel"
                    location_cache[cache_key] = location_id
                    return location_id
            # If no city, use first result
            location_id = data[0].get("dest_id")
            cache_key = f"{city_str}_hotel"
            location_cache[cache_key] = location_id
            return location_id
    except Exception as e:
        print(f"📍 Error using locations API for {simple_name}: {str(e)}")

    # Final fallback
    print(f"⚠️ Using default destination ID for {city_str}")
    return "-2092174"  # Default fallback

# === LANGCHAIN TOOLS ===

class ParseUserInputSchema(BaseModel):
    user_input: str = Field(description="The user's travel request text")

class ParseUserInputTool(BaseTool):
    name: str = "parse_user_input"
    description: str = "Extract travel details from user input text"
    args_schema: type[BaseModel] = ParseUserInputSchema
    
    def _run(self, user_input: str) -> Dict[str, Any]:
        """Extract travel details from user input text."""
        prompt = f'''
        Extract the following travel details from this text in JSON:
        - departure_city
        - destination_city
        - departure_date (YYYY-MM-DD)
        - return_date (YYYY-MM-DD, if any)
        - number_of_passengers
        - Hotel check-in date, use departure date if not provided
        - Hotel check-out date, use return date if not provided
        - for destination city and departure city add State and Country. Eg. Mumbai, Maharashtra, India.

        Text: "{user_input}"
        '''
        response = llm.invoke(prompt)
        content = response.content.strip()

        # Handle different JSON formats from LLM response
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0].strip()
        elif "```" in content:
            content = content.split("```")[1].strip()

        try:
            parsed_data = json.loads(content)
            # Normalize key names with consistent formatting
            normalized_data = {}
            for key, value in parsed_data.items():
                normalized_key = key.lower().replace(" ", "_")
                normalized_data[normalized_key] = value
            return normalized_data
        except json.JSONDecodeError:
            # Fallback for parsing errors
            return {
                "departure_city": None,
                "destination_city": None,
                "departure_date": None,
                "return_date": None,
                "number_of_passengers": 1
            }

class SearchFlightsSchema(BaseModel):
    departure_city: str = Field(description="City of departure")
    destination_city: str = Field(description="Destination city")
    departure_date: str = Field(description="Departure date (YYYY-MM-DD)")
    return_date: Optional[str] = Field(description="Return date (YYYY-MM-DD, optional)", default=None)
    number_of_passengers: int = Field(description="Number of passengers", default=1)

class SearchFlightsTool(BaseTool):
    name: str = "search_flights"
    description: str = "Search for flights between two cities"
    args_schema: type[BaseModel] = SearchFlightsSchema
    
    def _run(self, departure_city: str, destination_city: str, 
             departure_date: str, return_date: Optional[str] = None, 
             number_of_passengers: int = 1) -> Dict[str, Any]:
        """Searches for the cheapest flights using the Booking.com API with optimized location ID fetching."""
        # Handle invalid or missing inputs
        if not departure_city or not destination_city or not departure_date:
            return {"error": "Missing required flight search parameters"}

        # Get location IDs (using cache when available)
        from_id = get_city_location_id(departure_city, api_type="flight")
        to_id = get_city_location_id(destination_city, api_type="flight")

        if not from_id or not to_id:
            return {"error": f"Could not resolve location IDs for {departure_city} or {destination_city}"}

        url = "https://booking-com15.p.rapidapi.com/api/v1/flights/searchFlights"
        query = {
            "fromId": from_id,
            "toId": to_id,
            "departDate": departure_date,
            "returnDate": return_date or "",
            "adults": str(number_of_passengers),
            "children": "",
            "cabinClass": "ECONOMY",
            "currency_code": "INR",
            "stops": "none",
            "pageNo": "1",
            "sort": "CHEAPEST"
        }

        headers = {
            "X-RapidAPI-Key": RAPIDAPI_KEY,
            "X-RapidAPI-Host": "booking-com15.p.rapidapi.com"
        }

        try:
            res = requests.get(url, headers=headers, params=query, timeout=35)
            res.raise_for_status()
            return res.json()
        except Exception as e:
            print(f"🔍 Flight search error: {str(e)}")
            return {"error": f"Flight search failed: {str(e)}"}

class SearchHotelsSchema(BaseModel):
    destination_city: str = Field(description="Destination city")
    check_in_date: str = Field(description="Check-in date (YYYY-MM-DD)")
    check_out_date: Optional[str] = Field(description="Check-out date (YYYY-MM-DD, optional)", default=None)
    number_of_adults: int = Field(description="Number of adults", default=1)

class SearchHotelsTool(BaseTool):
    name: str = "search_hotels"
    description: str = "Search for hotels in a city"
    args_schema: type[BaseModel] = SearchHotelsSchema
    
    def _run(self, destination_city: str, check_in_date: str, 
             check_out_date: Optional[str] = None, number_of_adults: int = 1) -> Dict[str, Any]:
        """Searches for hotels using the Booking.com API with optimized location ID fetching."""
        if not destination_city or not check_in_date:
            return {"error": "Missing required hotel search parameters"}

        # Get location ID (using cache and optimized strategy)
        dest_id = get_hotel_location_id(destination_city)

        # Ensure check_out_date is set if not provided
        if not check_out_date:
            # Add a day to check_in_date for a default 1-night stay
            try:
                check_in_dt = datetime.strptime(check_in_date, "%Y-%m-%d")
                check_out_dt = check_in_dt + timedelta(days=1)
                check_out_date = check_out_dt.strftime("%Y-%m-%d")
            except ValueError:
                # Fallback if date parsing fails
                check_out_date = check_in_date

        print(f"Using destination ID: {dest_id}")
        url = "https://booking-com15.p.rapidapi.com/api/v1/hotels/searchHotels"
        querystring = {
            "dest_id": dest_id,
            "search_type": "CITY",
            "arrival_date": check_in_date,
            "departure_date": check_out_date,
            "adults": str(number_of_adults),
            "room_qty": "1",
            "page_number": "1",
            "units": "metric",
            "currency_code": "INR"
        }

        headers = {
            "X-RapidAPI-Key": RAPIDAPI_KEY,
            "X-RapidAPI-Host": "booking-com15.p.rapidapi.com"
        }

        try:
            print(f"Searching hotels with params: {querystring}")
            response = requests.get(url, headers=headers, params=querystring, timeout=30)
            response.raise_for_status()
            result = response.json()
            print(f"Hotel search completed successfully with status: {result.get('status', 'unknown')}")
            return result
        except Exception as e:
            print(f"🏨 Hotel search error: {str(e)}")
            return {"error": f"Hotel search failed: {str(e)}"}

# === HELPER FUNCTIONS ===

def format_datetime(dt_str):
    try:
        if not dt_str:
            return "N/A"
        return datetime.fromisoformat(dt_str.replace('Z', '+00:00')).strftime("%b %d, %Y at %I:%M %p")
    except Exception:
        return dt_str

def format_flight_results(flight_data):
    """Format flight search results into a readable summary."""
    if flight_data.get("error"):
        return f"❌ {flight_data['error']}"

    offers = flight_data.get("data", {}).get("flightOffers", [])
    if not offers:
        return "❌ No flights found or unexpected response format."

    seen = set()
    options = []

    for offer in offers:
        try:
            segments = offer.get("segments", [])
            if not segments:
                continue

            segment = segments[0]
            legs = segment.get("legs", [])
            if not legs:
                continue

            leg = legs[0]
            origin = leg.get("departureAirport", {}).get("name", "Unknown")
            destination = leg.get("arrivalAirport", {}).get("name", "Unknown")
            departure = format_datetime(leg.get("departureTime", ""))
            arrival = format_datetime(leg.get("arrivalTime", ""))

            carriers_data = leg.get("carriersData", [])
            airline = carriers_data[0].get("name", "Unknown") if carriers_data else "Unknown"

            price_info = offer.get("priceBreakdown", {}).get("totalRounded", {})
            units = price_info.get("units")
            nanos = price_info.get("nanos", 0)

            if units is not None:
                total_price = float(units) + float(nanos) / 1e9
                price_formatted = f"₹{int(round(total_price)):,}"
            else:
                total_price = float('inf')
                price_formatted = "₹N/A"

            flight_signature = (departure, arrival, airline, total_price)
            if flight_signature in seen:
                continue
            seen.add(flight_signature)

            options.append({
                "origin": origin,
                "destination": destination,
                "departure": departure,
                "arrival": arrival,
                "airline": airline,
                "price": total_price,
                "price_formatted": price_formatted
            })
        except Exception as e:
            print(f"Error processing flight offer: {str(e)}")
            continue

    if not options:
        return "❌ No valid flight options found."

    top_options = sorted(options, key=lambda x: x["price"])[:3]
    result = "\n📋 Top 3 Cheapest Flight Options:\n"
    
    for i, flight in enumerate(top_options, 1):
        result += f"\n✈️ Option {i}:\n"
        result += f"📍 From: {flight['origin']}\n"
        result += f"📍 To: {flight['destination']}\n"
        result += f"🕑 Departure: {flight['departure']}\n"
        result += f"🛬 Arrival: {flight['arrival']}\n"
        result += f"🛫 Airline: {flight['airline']}\n"
        result += f"💰 Price: {flight['price_formatted']}\n"
        result += "--------------------------------------------------------------\n"
    
    return result

def format_hotel_results(hotels):
    """Format hotel search results into a readable summary."""
    if hotels.get("error"):
        return f"❌ {hotels['error']}"

    if not hotels or not isinstance(hotels, dict):
        return "❌ Invalid hotel data format."

    data = hotels.get("data", {})
    if isinstance(data, dict):
        hotel_data = data.get("hotels", []) or data.get("results", [])
        if not hotel_data and "data" in data:
            inner_data = data.get("data", {})
            if isinstance(inner_data, dict):
                hotel_data = inner_data.get("hotels", []) or inner_data.get("results", [])
    else:
        hotel_data = []

    if not hotel_data:
        return "❌ No hotels found in the response."

    options = []
    for hotel in hotel_data:
        try:
            property_info = hotel.get("property", {})
            name = property_info.get("name", "Unknown")

            # Extract price from accessibilityLabel
            accessibility = hotel.get("accessibilityLabel", "")
            price = "N/A"
            match_price = re.search(r"Current price (\d+)", accessibility)
            if match_price:
                price = f"₹{match_price.group(1)}"

            # Address fallback using latitude and country code
            latitude = property_info.get("latitude", "")
            country = property_info.get("countryCode", "").upper()
            address = f"Lat: {latitude}, Country: {country}" if latitude else "N/A"

            # Extract rating from accessibilityLabel
            rating = "N/A"
            match_rating = re.search(r"(\d+(\.\d+)?) Very good", accessibility)
            if match_rating:
                rating = match_rating.group(1)

            options.append({
                "name": name,
                "price": price,
                "address": address,
                "rating": rating
            })
        except Exception as e:
            print(f"Error processing hotel data: {str(e)}")
            continue

    if not options:
        return "❌ No valid hotel options found."

    # Sort by price
    def price_sort_key(hotel):
        price_str = hotel.get("price", "N/A")
        if isinstance(price_str, (int, float)):
            return price_str
        try:
            numeric_part = ''.join(c for c in price_str if c.isdigit())
            return float(numeric_part) if numeric_part else float('inf')
        except:
            return float('inf')

    sorted_options = sorted(options, key=price_sort_key)[:3]

    result = "\n🏨 Top 3 Hotel Options:\n"
    
    for i, hotel in enumerate(sorted_options, 1):
        result += f"\n🏨 Option {i}:\n"
        result += f"🏨 Hotel: {hotel['name']}\n"
        result += f"💰 Price: {hotel['price']}\n"
        result += f"📍 Address: {hotel['address']}\n"
        result += f"⭐ Rating: {hotel['rating']}\n"
        result += "--------------------------------------------------------------\n"
    
    return result

# === LANGGRAPH NODES ===

def parse_request(state: TravelState) -> TravelState:
    """Parse the user request to extract travel details."""
    # Get the last user message
    for message in reversed(state["messages"]):
        if message["role"] == "human":
            user_input = message["content"]
            break
    else:
        return {**state, "error": "No user message found", "next": "respond"}
    
    try:
        parse_tool = ParseUserInputTool()
        travel_details = parse_tool._run(user_input=user_input)
        
        # Update the state with parsed details
        return {
            **state,
            "travel_details": travel_details,
            "next": "check_details"
        }
    except Exception as e:
        return {**state, "error": f"Failed to parse request: {str(e)}", "next": "respond"}

def check_details(state: TravelState) -> TravelState:
    """Check if we have enough details to proceed with search."""
    details = state["travel_details"]
    missing = []
    
    if not details.get("departure_city"):
        missing.append("departure city")
    if not details.get("destination_city"):
        missing.append("destination city")
    if not details.get("departure_date"):
        missing.append("departure date")
        
    if missing:
        missing_str = ", ".join(missing)
        return {
            **state,
            "error": f"Missing required travel details: {missing_str}",
            "next": "respond"
        }
    
    return {**state, "next": "search_flights"}

def search_flights_node(state: TravelState) -> TravelState:
    """Search for flights based on travel details."""
    details = state["travel_details"]
    
    try:
        flight_tool = SearchFlightsTool()
        flight_data = flight_tool._run(
            departure_city=details.get("departure_city"),
            destination_city=details.get("destination_city"),
            departure_date=details.get("departure_date"),
            return_date=details.get("return_date"),
            number_of_passengers=details.get("number_of_passengers", 1)
        )
        
        return {
            **state,
            "flight_data": flight_data,
            "next": "search_hotels"
        }
    except Exception as e:
        return {**state, "error": f"Flight search failed: {str(e)}", "next": "respond"}

def search_hotels_node(state: TravelState) -> TravelState:
    """Search for hotels based on travel details."""
    details = state["travel_details"]
    
    try:
        hotel_tool = SearchHotelsTool()
        hotel_data = hotel_tool._run(
            destination_city=details.get("destination_city"),
            check_in_date=details.get("hotel_check-in_date", details.get("departure_date")),
            check_out_date=details.get("hotel_check-out_date", details.get("return_date")),
            number_of_adults=details.get("number_of_passengers", 1)
        )
        
        return {
            **state,
            "hotel_data": hotel_data,
            "next": "prepare_itinerary"
        }
    except Exception as e:
        return {**state, "error": f"Hotel search failed: {str(e)}", "next": "respond"}

def prepare_itinerary(state: TravelState) -> TravelState:
    """Prepare a complete itinerary with flight and hotel options."""
    flight_data = state.get("flight_data", {})
    hotel_data = state.get("hotel_data", {})
    
    flight_summary = format_flight_results(flight_data)
    hotel_summary = format_hotel_results(hotel_data)
    
    itinerary = {
        "flight_summary": flight_summary,
        "hotel_summary": hotel_summary,
        "total_summary": f"{flight_summary}\n\n{hotel_summary}"
    }
    
    return {**state, "itinerary": itinerary, "next": "respond"}

def generate_response(state: TravelState) -> TravelState:
    """Generate a response to the user based on the current state."""
    if state.get("error"):
        response = f"I'm sorry, but there was an issue with your request: {state['error']}\n\n"
        response += "Please provide more details so I can help you book your travel."
    elif state.get("itinerary"):
        itinerary = state["itinerary"]
        response = f"I've found some great options for your trip!\n\n{itinerary['total_summary']}"
        response += "\nWould you like me to help you refine these options or provide more information about any of these choices?"
    else:
        response = "I'm processing your travel request. Could you provide more details about your trip?"
    
    # Add the response to messages
    new_messages = state["messages"] + [{"role": "ai", "content": response}]
    
    return {**state, "messages": new_messages, "next": END}

# === BUILD THE GRAPH ===

def build_travel_agent_graph():
    """Build the LangGraph for the travel booking agent."""
    # Define the nodes
    nodes = {
        "parse_request": parse_request,
        "check_details": check_details,
        "search_flights": search_flights_node,
        "search_hotels": search_hotels_node,
        "prepare_itinerary": prepare_itinerary,
        "respond": generate_response
    }
    
    # Build the graph
    workflow = StateGraph(TravelState)
    
    # Add nodes
    for name, fn in nodes.items():
        workflow.add_node(name, fn)
    
    # Add conditional edges
    workflow.add_conditional_edges(
        "parse_request",
        lambda x: x["next"],
        {
            "check_details": "check_details",
            "respond": "respond"
        }
    )
    
    workflow.add_conditional_edges(
        "check_details",
        lambda x: x["next"],
        {
            "search_flights": "search_flights",
            "respond": "respond"
        }
    )
    
    workflow.add_conditional_edges(
        "search_flights",
        lambda x: x["next"],
        {
            "search_hotels": "search_hotels",
            "respond": "respond"
        }
    )
    
    workflow.add_conditional_edges(
        "search_hotels",
        lambda x: x["next"],
        {
            "prepare_itinerary": "prepare_itinerary",
            "respond": "respond"
        }
    )
    
    workflow.add_edge("prepare_itinerary", "respond")
    workflow.add_edge("respond", END)
    
    # Set the entry point
    workflow.set_entry_point("parse_request")
    
    return workflow.compile()

# === MAIN APPLICATION ===

def run_travel_agent():
    """Run the travel agent workflow with a conversational interface."""
    # Build the workflow
    travel_agent_graph = build_travel_agent_graph()
    
    print("🌍 Travel Booking Agent powered by LangGraph")
    print("--------------------------------------------")
    print("Example: 'I need a flight from Mumbai to Delhi on 2025-06-10 returning on 2025-06-15 for 2 passengers'")
    
    # Initialize conversation state
    state = {
        "messages": [],
        "travel_details": {},
        "flight_data": None,
        "hotel_data": None,
        "itinerary": None,
        "error": None,
        "next": None
    }
    
    while True:
        user_input = input("\n💬 You: ")
        if user_input.lower() in ["exit", "quit", "bye"]:
            print("👋 Thank you for using the Travel Booking Agent. Goodbye!")
            break
            
        # Add user message to state
        state["messages"].append({"role": "human", "content": user_input})
        
        # Run the graph
        result = travel_agent_graph.invoke(state)
        
        # Update state for next iteration
        state = result.copy()
        
        # Print AI response
        ai_messages = [msg for msg in state["messages"] if msg["role"] == "ai"]
        if ai_messages:
            print(f"\n🤖 Travel Agent: {ai_messages[-1]['content']}")

if __name__ == "__main__":
    run_travel_agent()
