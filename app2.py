import streamlit as st
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain_community.llms import Ollama
from datetime import datetime, timedelta

# Configure page
st.set_page_config(
    page_title="Tourism Chatbot EG",
    page_icon="🌍",
    layout="wide"
)

# Title with better styling
st.markdown("""
    <h1 style='text-align: center; color: #2E86C1;'>
        Tourism Assistant
    </h1>
""", unsafe_allow_html=True)

# Structured knowledge base with keywords and responses
EGYPT_TOURISM_KB = [
    {
        "keywords": ["famous", "tourist", "attractions", "popular", "places", "visit", "sites", "landmarks", "see"],
        "query_must_include": ["egypt"],
        "response": "The most famous tourist attractions in Egypt are the Pyramids of Giza, the Sphinx, the Egyptian Museum, Luxor and Karnak Temples, the Valley of the Kings, Abu Simbel temples, and Nile River cruises."
    },
    {
        "keywords": ["safe", "safety", "danger", "dangerous", "secure"],
        "query_must_include": ["travel", "egypt", "visit"],
        "response": "Yes, Egypt is generally safe for tourists, especially in popular tourist areas. However, it's always recommended to stay vigilant, avoid isolated areas at night, and follow local advice."
    },
    {
        "keywords": ["visa", "need visa", "travel document"],
        "query_must_include": ["visa"],
        "response": "Most travelers need a visa to visit Egypt, which can often be obtained on arrival or online via e-Visa."
    },
    {
        "keywords": ["best time", "when visit", "season", "weather", "when to go"],
        "query_must_include": ["time", "when", "season", "visit"],
        "response": "The best time to visit Egypt is from October to April, when the weather is cooler and ideal for sightseeing. Summer months (May to September) can be extremely hot, especially in Upper Egypt."
    },
    {
        "keywords": ["language", "speak", "official", "arabic"],
        "query_must_include": ["language", "speak"],
        "response": "Arabic is the official language in Egypt, but English is widely spoken in tourist areas, hotels, and restaurants."
    },
    {
        "keywords": ["currency", "money", "cash", "egyptian pound", "pound"],
        "query_must_include": ["currency", "money", "cash"],
        "response": "The Egyptian Pound (EGP) is the official currency used in Egypt."
    },
    {
        "keywords": ["us dollars", "euros", "foreign currency", "exchange"],
        "query_must_include": ["dollar", "euro", "currency"],
        "response": "Some places in Egypt accept US dollars or Euros, but it's better to use local currency. Currency exchange is available at banks, hotels, and authorized exchange offices."
    },
    {
        "keywords": ["credit card", "card", "pay", "payment", "mastercard", "visa card"],
        "query_must_include": ["card", "credit", "pay"],
        "response": "Credit cards are accepted in most hotels, restaurants, and shops in major cities in Egypt. However, it's advisable to carry some cash for smaller establishments and markets."
    },
    {
        "keywords": ["historical", "sites", "cairo", "history"],
        "query_must_include": ["cairo", "historical", "history", "sites"],
        "response": "The top historical sites in Cairo include the Pyramids of Giza, the Great Sphinx, the Egyptian Museum, Salah El-Din Citadel, Khan el-Khalili bazaar, Old Cairo, and various mosques and palaces."
    },
    {
        "keywords": ["nile", "cruise", "boat", "river", "sail"],
        "query_must_include": ["nile", "cruise", "boat"],
        "response": "Yes, Nile cruises between Luxor and Aswan are very popular and a wonderful way to experience Egypt's ancient temples along the river. Most cruises range from 3 to 7 nights."
    },
    {
        "keywords": ["tip", "tipping", "baksheesh", "service", "how much tip"],
        "query_must_include": ["tip", "tipping"],
        "response": "Yes, tipping (called 'baksheesh') is common and appreciated in Egypt. Typically, 10% in restaurants (if service charge isn't included), 5-10 EGP for porters, and 10-20 EGP for tour guides per day is appropriate."
    },
    {
        "keywords": ["food", "eat", "cuisine", "traditional", "dishes", "meal"],
        "query_must_include": ["food", "eat", "cuisine", "dish"],
        "response": "Traditional Egyptian food includes koshari (a mix of rice, pasta, and lentils), ful medames (stewed fava beans), ta'ameya (Egyptian falafel), molokheya (jute leaf stew), grilled kofta and kebabs, and various mezze dishes. Don't miss trying Egyptian desserts like baklava and kunafa."
    },
    {
        "keywords": ["water", "drink", "tap", "bottled", "safe water"],
        "query_must_include": ["water", "drink"],
        "response": "It's not recommended to drink tap water in Egypt. Stick to bottled water, which is widely available and inexpensive."
    },
    {
        "keywords": ["beach", "beaches", "swim", "swimming", "sea", "coast"],
        "query_must_include": ["beach", "beaches", "sea"],
        "response": "Yes, Egypt has beautiful beaches, especially along the Red Sea in resorts like Sharm El Sheikh, Hurghada, Marsa Alam, and Dahab. These areas are famous for their clear waters, coral reefs, and excellent diving opportunities."
    },
    {
        "keywords": ["family", "children", "kids", "child", "friendly"],
        "query_must_include": ["family", "children", "kids", "child"],
        "response": "Egypt is good for family travel with many family-friendly destinations including the historical sites, beach resorts, and desert safaris. Many hotels offer children's activities and facilities."
    },
    {
        "keywords": ["english", "speak english", "language barrier"],
        "query_must_include": ["english", "speak"],
        "response": "Many people in the tourism industry in Egypt speak basic to good English, especially in hotels, restaurants, and popular tourist sites. Communication is generally not an issue for English speakers."
    },
    {
        "keywords": ["clothes", "dress", "wear", "appropriate", "conservative"],
        "query_must_include": ["clothes", "dress", "wear"],
        "response": "Light, modest clothing is best for Egypt, especially in religious areas. Women should avoid revealing clothes in public areas. For visiting mosques, women should cover their shoulders, legs, and hair, and everyone should remove shoes."
    },
    {
        "keywords": ["internet", "wifi", "connection", "online"],
        "query_must_include": ["wifi", "internet", "online"],
        "response": "Most hotels, cafes, and restaurants in Egypt offer free Wi-Fi. You can also purchase a local SIM card with data for mobile connectivity, which is inexpensive and provides good coverage in major areas."
    },
    {
        "keywords": ["transportation", "travel", "get around", "transport", "move"],
        "query_must_include": ["transportation", "transport", "travel", "get around"],
        "response": "Transportation options in Egypt include taxis, Uber (in Cairo and Alexandria), buses, trains (including sleeper trains between Cairo and Luxor/Aswan), and domestic flights. For shorter distances within cities, taxis or ride-sharing services are recommended."
    },
    {
        "keywords": ["inside", "enter", "pyramid", "pyramids", "interior"],
        "query_must_include": ["pyramid", "inside", "enter"],
        "response": "Yes, you can visit the inside of some pyramids in Egypt, but it requires an extra ticket and space is limited. The main accessible ones are the Great Pyramid of Khufu, the Pyramid of Khafre, and the Bent and Red Pyramids at Dahshur. Be aware that the passages are narrow and can be hot and cramped."
    },
    {
        "keywords": ["time zone", "clock", "time difference"],
        "query_must_include": ["time zone", "time"],
        "response": "Egypt's time zone is GMT+2 (Eastern European Time)."
    },
    {
        "keywords": ["alone", "solo", "single", "traveler", "woman"],
        "query_must_include": ["alone", "solo", "single"],
        "response": "It's generally safe to travel alone in Egypt, but always stay in tourist-friendly areas and follow local advice. Solo female travelers should be aware that they may receive unwanted attention and should dress modestly. Consider joining guided tours for more remote areas."
    },
    {
        "keywords": ["souvenirs", "buy", "shopping", "gift", "market"],
        "query_must_include": ["souvenir", "buy", "shop"],
        "response": "Popular souvenirs from Egypt include papyrus art, Egyptian cotton products, perfumes, spices, jewelry, alabaster items, hand-blown glass, and miniature pyramids and sphinx figures. Khan el-Khalili bazaar in Cairo is a famous shopping destination."
    },
    {
        "keywords": ["headscarf", "hijab", "head covering", "veil", "scarf"],
        "query_must_include": ["headscarf", "hijab", "scarf"],
        "response": "Women generally don't need to wear a headscarf in Egypt, but it's respectful to carry one when visiting mosques or religious sites. In tourist areas and major cities, western clothing is common."
    },
    {
        "keywords": ["desert", "safari", "western desert", "white desert", "black desert", "siwa"],
        "query_must_include": ["desert", "safari"],
        "response": "Egypt has amazing deserts to explore, including the White Desert with its unique chalk formations, the Black Desert, Siwa Oasis, and the Western Desert. Desert safaris, camping under the stars, and visits to oases are popular activities."
    },
    {
        "keywords": ["camel", "ride", "camel ride", "pyramid"],
        "query_must_include": ["camel", "ride"],
        "response": "Yes, camel rides are a popular tourist activity near the pyramids in Giza and in desert areas. Negotiate the price before mounting, and a typical ride might cost between 100-200 EGP."
    },
    {
        "keywords": ["cash", "money", "atm", "exchange", "carry"],
        "query_must_include": ["cash", "money", "carry"],
        "response": "In Egypt, carry small amounts of cash for tips and street shopping, and use cards when possible. ATMs are widely available in cities and tourist areas. It's advisable to have some small denominations for tipping and small purchases."
    },
    {
        "keywords": ["electrical", "plugs", "socket", "voltage", "adapter"],
        "query_must_include": ["electrical", "plug", "socket", "voltage"],
        "response": "Egypt uses Type C and F electrical plugs with a 220V supply. Travelers from countries using different plug types or voltage should bring an appropriate adapter."
    },
    {
        "keywords": ["public", "transportation", "metro", "bus", "train", "cairo"],
        "query_must_include": ["public", "transportation", "metro", "bus"],
        "response": "Cairo has a metro system and buses, but taxis are often easier for tourists. The metro is clean, efficient, and very affordable, with separate cars for women. Buses can be crowded and challenging to navigate for tourists."
    },
    {
        "keywords": ["tour", "guide", "guided", "group"],
        "query_must_include": ["tour", "guide", "guided"],
        "response": "There are many tour companies in Egypt offering guided trips to historical sites. Having a knowledgeable guide can greatly enhance your understanding of Egypt's history and culture. Licensed guides can be arranged through hotels or reputable tour agencies."
    },
    {
        "keywords": ["security", "robbery", "robbed", "theft", "stolen", "emergency"],
        "query_must_include": ["security", "robbery", "robbed", "theft", "stolen", "emergency"],
        "response": "If you get robbed in Egypt: 1) Report to the Tourist Police immediately (they have offices in all major tourist areas and speak English). 2) Contact your embassy or consulate. 3) File a police report for insurance purposes. 4) Cancel any stolen credit cards or IDs. 5) Always keep emergency contacts handy including your country's embassy number."
    }
]

# Initialize session states
if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferMemory(
        return_messages=True,
        memory_key="history"
    )

if "messages" not in st.session_state:
    st.session_state.messages = []

if "last_request" not in st.session_state:
    st.session_state.last_request = datetime.now() - timedelta(seconds=10)

if "cache" not in st.session_state:
    st.session_state.cache = {}

# Set deepseek-llm as the default model
model_name = "deepseek-llm"

# Set Ollama base URL
ollama_base_url = st.sidebar.text_input(
    "Ollama API URL",
    value="https://e50a-156-199-172-163.ngrok-free.app",
    help="Enter the URL where Ollama is running"
)


# Improved function to find matching context from knowledge base
def find_context_match(query, min_score=0.6):
    """
    Find the best matching response for a query from the knowledge base
    using a more sophisticated matching algorithm
    """
    if not query:
        return None, 0

    query = query.lower().strip()
    best_match = None
    best_score = 0

    for entry in EGYPT_TOURISM_KB:
        # Check if any required keywords are present
        must_include = entry.get("query_must_include", [])
        has_required = any(word in query for word in must_include) if must_include else True

        if not has_required:
            continue

        # Score the match based on keyword presence
        match_keywords = entry["keywords"]
        keyword_matches = sum(1 for keyword in match_keywords if keyword in query)

        # Calculate keyword coverage
        if keyword_matches > 0:
            # Proportional score based on matched keywords and query relevance
            query_words = query.split()
            query_word_count = len(query_words)

            # Calculate how much of the query is covered by keywords
            covered_words = sum(
                1 for word in query_words if any(keyword in word or word in keyword for keyword in match_keywords))
            coverage_score = covered_words / query_word_count if query_word_count > 0 else 0

            # Calculate keyword match ratio
            keyword_score = keyword_matches / len(match_keywords)

            # Combined score with emphasis on coverage
            score = (coverage_score * 0.7) + (keyword_score * 0.3)

            if score > best_score and score >= min_score:
                best_score = score
                best_match = entry["response"]

    return best_match, best_score


@st.cache_resource
def get_llm(temperature=0.7):
    """Initialize LLM with caching to prevent reinitialization"""
    try:
        return Ollama(
            model=model_name,
            base_url=ollama_base_url,
            temperature=temperature
        )
    except Exception as e:
        st.error(f"Error initializing model: {str(e)}")
        return None


# Initialize the language model with enhanced error handling
try:
    # Get temperature from sidebar
    temperature = st.sidebar.slider("Temperature", min_value=0.0, max_value=1.0, value=0.7, step=0.1,
                                    help="Higher values make output more random, lower values make it more deterministic")

    # Use cached LLM
    llm = get_llm(temperature)
    if llm is None:
        raise Exception("Failed to initialize language model")

    # Create a custom prompt template with context awareness
    template = """The following is a conversation between a human and an AI tourism assistant.
    You specialize in providing information about travel destinations, particularly Egypt.
    Give helpful, accurate, and concise answers.

    {history}
    Human: {input}
    AI:"""

    prompt = PromptTemplate(
        input_variables=["history", "input"],
        template=template
    )

    conversation = ConversationChain(
        llm=llm,
        memory=st.session_state.memory,
        prompt=prompt,
        verbose=False
    )

    st.success(f"Model connected successfully! Using: {model_name}")

except Exception as e:
    error_msg = str(e)
    st.error(f"Error initializing model: {error_msg}")

    st.warning("""
    Make sure Ollama is installed and running:
    1. Install Ollama from https://ollama.ai/
    2. Start the Ollama server
    3. Pull the model using: `ollama pull deepseek-llm`
    """)
    st.stop()

# Display chat history with improved formatting
for message in st.session_state.messages:
    avatar = "🧑" if message["role"] == "user" else "🌍"
    with st.chat_message(message["role"], avatar=avatar):
        st.markdown(message["content"])

# Enhanced rate limiting
if (datetime.now() - st.session_state.last_request).seconds < 1:  # Reduced from 2 to 1 second
    st.warning("Please wait a moment before sending another message.")
    st.stop()

# User input
prompt = st.chat_input("Ask about destinations, culture, or safety tips...")

if prompt:
    st.session_state.last_request = datetime.now()
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user", avatar="🧑"):
        st.markdown(prompt)

    with st.chat_message("assistant", avatar="🌍"):
        with st.spinner("Generating response..."):
            try:
                # Check cache first
                cache_key = prompt.lower()
                if cache_key in st.session_state.cache:
                    response = st.session_state.cache[cache_key]
                else:
                    # Check knowledge base for quick response
                    kb_response, match_score = find_context_match(prompt)

                    # If we have a good match from the knowledge base
                    if kb_response and match_score > 0.6:
                        response = kb_response
                    else:
                        # No good match - use the LLM
                        full_prompt = f"""You are a tourism assistant specializing in Egypt travel.
                        Question: {prompt}

                        Consider these categories for your answer:
                        - Historical sites and attractions
                        - Cultural information
                        - Practical travel advice
                        - Safety information
                        - Local customs

                        Answer concisely but informatively."""

                        response = conversation.predict(input=full_prompt)

                    # Store in cache
                    st.session_state.cache[cache_key] = response

                # Display cleaned response
                cleaned_response = response.strip()
                st.markdown(cleaned_response)
                st.session_state.messages.append({"role": "assistant", "content": cleaned_response})

            except Exception as e:
                error_response = f"Sorry, I encountered an error: {str(e)}. Please try again later."
                st.error(error_response)
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": error_response
                })

# Sidebar with deployment-ready information
with st.sidebar:
    st.header("ℹ️ About")

    st.markdown("""
    **Tourism Expert Chatbot**  
    • Powered by DeepSeek-LLM (local)  
    • Provides detailed travel information  
    • Offers cultural insights  
    • Remembers conversation history  
    • Fast responses for common questions
    """)

    # Add clear cache button
    if st.button("Clear Response Cache"):
        st.session_state.cache = {}
        st.success("Cache cleared!")

    # Add reset button to clear conversation
    if st.button("Reset Conversation"):
        st.session_state.messages = []
        st.session_state.memory.clear()
        st.experimental_rerun()

    # Add model management section
    st.subheader("Model Settings")

    # Add button to pull model if not already downloaded
    if st.button("Pull DeepSeek-LLM Model"):
        import subprocess

        try:
            st.info("Pulling deepseek-llm model... This may take a while.")
            subprocess.run(["ollama", "pull", "deepseek-llm"], check=True)
            st.success("Successfully pulled deepseek-llm model!")
        except Exception as e:
            st.error(f"Error pulling model: {str(e)}")
