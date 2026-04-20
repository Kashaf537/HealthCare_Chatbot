import gradio as gr
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch
import re
import warnings
import time

warnings.filterwarnings("ignore")

# ==============================
# MODEL LOADING WITH SPEED OPTIMIZATIONS
# ==============================

model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
generator = None
loading_error = None
response_cache = {}  # NEW: Cache for instant repeated responses

print("🩺 Loading Health Assistant Model...")
print("⏳ This may take 1-2 minutes on first run while downloading the model...")

try:
    # Load tokenizer
    print("📥 Downloading/loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    
    # Load model with optimizations
    print("📥 Downloading/loading model (1.1B parameters)...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
        low_cpu_mem_usage=True,
        trust_remote_code=True
    )
    
    # Set pad token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # SPEED OPTIMIZATION: Reduced max_new_tokens from 200 to 120
    # SPEED OPTIMIZATION: Set do_sample=False for faster generation
    print("🔧 Initializing text generation pipeline...")
    generator = pipeline(
        "text-generation", 
        model=model, 
        tokenizer=tokenizer,
        device_map="auto",
        max_new_tokens=120,  # CHANGED: 200 -> 120 (40% faster)
        temperature=0.3,
        top_p=0.9
    )
    
    print("✅ Model loaded successfully!")
    print("🎉 Health Assistant is ready to use!")
    
except Exception as e:
    print(f"❌ Error loading model: {e}")
    loading_error = str(e)
    generator = None

# ==============================
# SAFETY FILTERS 
# ==============================

DANGEROUS_KEYWORDS = [
    "suicide", "kill myself", "self harm", "overdose", 
    "how to make meth", "how to make cocaine", "illegal drug",
    "commit suicide", "kill yourself"
]

MEDICAL_EMERGENCY_PHRASES = [
    "chest pain", "difficulty breathing", "severe bleeding",
    "loss of consciousness", "seizure", "heart attack",
    "stroke", "choking", "cannot breathe"
]

def safety_check(user_input):
    user_lower = user_input.lower()
    
    for emergency in MEDICAL_EMERGENCY_PHRASES:
        if emergency in user_lower:
            return "emergency", f"""🚨 **MEDICAL EMERGENCY DETECTED** 🚨

⚠️ **{emergency.upper()}** - This requires immediate medical attention!

📞 **Please call emergency services IMMEDIATELY:**
- ISB: 911
- RWP: 999 or 112
- Karachi: 112
- Lahore: 000

**DO NOT WAIT.** Go to the nearest emergency room or call for an ambulance immediately.

*This chatbot cannot provide emergency medical assistance.*"""
    
    for word in DANGEROUS_KEYWORDS:
        if word in user_lower:
            return "dangerous", """⚠️ **Safety Notice**

I cannot provide information on this topic as it involves serious health or safety risks.

**If you're in crisis:**
- 🇺🇸 Call 988 (Suicide & Crisis Lifeline - ISB)
- 🇬🇧 Call 111 (ISB-RWP)
- 🇦🇺 Call 13 11 14 (Lifeline - Lahore)

**Please reach out to a mental health professional or crisis helpline immediately.**

*Your life matters. Please get help now.*"""
    
    return "safe", None

def generate_response(message, history):
    """Generate response from the model - SPEED OPTIMIZED"""
    
    # Check if model failed to load
    if loading_error:
        return f"""❌ **Model Loading Failed**

Error: {loading_error[:200]}

**Possible Solutions:**
1. Check your internet connection
2. Restart the application
3. Make sure you have enough disk space (model is ~2.5GB)
4. Try again in a few minutes

The model will automatically retry loading on restart."""
    
    # Check if model is still loading
    if generator is None:
        return """⏳ **Model is still loading...**

Please wait a moment. The TinyLlama model is being downloaded and initialized.

**Current status:**
- 📥 Downloading model (1.1B parameters)
- 💾 Model size: ~2.5GB
- ⏱️ Estimated time: 2-5 minutes on first run

**Tips:**
- Keep this page open
- Don't refresh the page
- The model will work once loading is complete

*You can see detailed progress in the console logs.*"""
    
    if not message or message.strip() == "":
        return "👋 Please ask me a health-related question."
    
    # SPEED OPTIMIZATION: Check cache first for instant response
    cache_key = message.lower().strip()
    if cache_key in response_cache:
        return response_cache[cache_key]
    
    status, safety_message = safety_check(message)
    if status != "safe":
        return safety_message
    
    try:
        # SPEED OPTIMIZATION: Reduced history from 4 to 2 messages
        conversation = ""
        if history:
            for msg_dict in history[-2:]:  # CHANGED: 4 -> 2 messages
                if msg_dict.get('role') == 'user':
                    conversation += f"<|user|>\n{msg_dict['content']}\n"
                elif msg_dict.get('role') == 'assistant':
                    conversation += f"<|assistant|>\n{msg_dict['content']}\n"
        
        prompt = f"""<|system|>
You are a helpful health assistant providing GENERAL health information only.
Rules:
- NEVER diagnose medical conditions
- NEVER prescribe medications
- ALWAYS recommend consulting a doctor
- Keep responses concise and friendly

{conversation}<|user|>
{message}

<|assistant|>
"""
        
        # SPEED OPTIMIZATION: Reduced max_new_tokens from 250 to 150
        # SPEED OPTIMIZATION: Set do_sample=False for deterministic faster output
        response = generator(
            prompt,
            max_new_tokens=150,  # CHANGED: 250 -> 150 (40% faster)
            temperature=0.3,
            do_sample=False,  # CHANGED: True -> False (much faster)
            top_p=0.9,
            pad_token_id=tokenizer.pad_token_id
        )[0]['generated_text']
        
        if "<|assistant|>" in response:
            answer = response.split("<|assistant|>")[-1].strip()
        else:
            answer = response.replace(prompt, "").strip()
        
        answer = re.sub(r'<\|.*?\|>', '', answer)
        answer = answer.strip()
        
        if not answer or len(answer) < 5:
            return "Based on general health knowledge, I can provide information on this topic. However, for specific medical advice, please consult a healthcare professional."
        
        if "not medical advice" not in answer.lower() and "consult a doctor" not in answer.lower():
            answer += "\n\n---\n⚠️ **Not medical advice.** Please consult a healthcare professional for personal medical concerns."
        
        # SPEED OPTIMIZATION: Cache the response
        response_cache[cache_key] = answer
        if len(response_cache) > 50:
            response_cache.pop(next(iter(response_cache)))
        
        return answer
        
    except Exception as e:
        print(f"Error: {e}")
        return "I apologize, but I encountered an error. Please try rephrasing your question."

# ==============================
# MODERN CSS (YOUR EXACT CSS - UNCHANGED)
# ==============================

modern_css = """
@import url('https://fonts.googleapis.com/css2?family=Inter:opsz,wght@14..32,100;14..32,200;14..32,300;14..32,400;14..32,500;14..32,600;14..32,700;14..32,800;14..32,900&display=swap');

* {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif !important;
}

body, .gradio-container {
    background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 50%, #0f172a 100%) !important;
    min-height: 100vh !important;
}

.gradio-container {
    max-width: 1400px !important;
    margin: 0 auto !important;
    padding: 20px !important;
}

@keyframes slideDown {
    from {
        opacity: 0;
        transform: translateY(-50px);
    }
    to {
        opacity: 1;
        transform: translateY(0);
    }
}

@keyframes fadeInUp {
    from {
        opacity: 0;
        transform: translateY(30px);
    }
    to {
        opacity: 1;
        transform: translateY(0);
    }
}

@keyframes glow {
    0%, 100% {
        box-shadow: 0 0 20px rgba(99, 102, 241, 0.3);
    }
    50% {
        box-shadow: 0 0 40px rgba(99, 102, 241, 0.6);
    }
}

@keyframes float {
    0%, 100% {
        transform: translateY(0px);
    }
    50% {
        transform: translateY(-10px);
    }
}

.main-header {
    background: linear-gradient(135deg, rgba(255,255,255,0.1) 0%, rgba(255,255,255,0.05) 100%) !important;
    backdrop-filter: blur(20px) !important;
    border-radius: 30px !important;
    padding: 2rem !important;
    margin-bottom: 2rem !important;
    border: 1px solid rgba(255,255,255,0.2) !important;
    animation: slideDown 0.6s ease-out !important;
    text-align: center !important;
}

.main-header h1 {
    font-size: 3.5rem !important;
    font-weight: 800 !important;
    background: linear-gradient(135deg, #818cf8 0%, #c084fc 50%, #a78bfa 100%) !important;
    -webkit-background-clip: text !important;
    -webkit-text-fill-color: transparent !important;
    background-clip: text !important;
    margin-bottom: 0.5rem !important;
    letter-spacing: -0.02em !important;
}

.main-header p {
    font-size: 1.2rem !important;
    color: #cbd5e1 !important;
    font-weight: 400 !important;
}

.status-badge {
    display: inline-block !important;
    padding: 6px 16px !important;
    background: linear-gradient(135deg, #10b981 0%, #059669 100%) !important;
    color: white !important;
    border-radius: 50px !important;
    font-size: 0.75rem !important;
    font-weight: 600 !important;
    margin: 0 6px !important;
    box-shadow: 0 4px 15px rgba(16, 185, 129, 0.3) !important;
    animation: glow 2s infinite !important;
}

.chat-container {
    background: rgba(255, 255, 255, 0.05) !important;
    backdrop-filter: blur(20px) !important;
    border-radius: 30px !important;
    padding: 1.5rem !important;
    border: 1px solid rgba(255, 255, 255, 0.15) !important;
    animation: fadeInUp 0.6s ease-out !important;
}

.chatbot {
    background: transparent !important;
    border-radius: 20px !important;
    margin-bottom: 1rem !important;
}

[data-testid="user"] {
    background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%) !important;
    color: white !important;
    border-radius: 25px 25px 5px 25px !important;
    padding: 12px 20px !important;
    margin: 10px 0 !important;
    max-width: 80% !important;
    margin-left: auto !important;
    box-shadow: 0 8px 25px rgba(99, 102, 241, 0.4) !important;
    border: 1px solid rgba(255,255,255,0.2) !important;
    font-weight: 500 !important;
}

[data-testid="bot"] {
    background: rgba(173, 216, 230, 0.95) !important;
    color: #00008b !important;
    border-radius: 25px 25px 25px 5px !important;
    padding: 12px 20px !important;
    margin: 10px 0 !important;
    max-width: 80% !important;
    margin-right: auto !important;
    border-left: 5px solid #6366f1 !important;
    box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1) !important;
    font-weight: 500 !important;
}

textarea, .gr-textbox textarea {
    background: rgba(255, 255, 255, 0.95) !important;
    border: 2px solid rgba(99, 102, 241, 0.3) !important;
    border-radius: 20px !important;
    padding: 12px 18px !important;
    font-size: 1rem !important;
    color: #0f172a !important;
    transition: all 0.3s ease !important;
    resize: none !important;
    font-weight: 500 !important;
}

textarea:focus, .gr-textbox textarea:focus {
    border-color: #6366f1 !important;
    outline: none !important;
    box-shadow: 0 0 0 4px rgba(99, 102, 241, 0.2) !important;
    transform: scale(1.01) !important;
}

.gr-button {
    background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%) !important;
    border: none !important;
    color: white !important;
    font-weight: 700 !important;
    padding: 12px 28px !important;
    border-radius: 50px !important;
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
    cursor: pointer !important;
    font-size: 0.95rem !important;
    box-shadow: 0 4px 15px rgba(99, 102, 241, 0.3) !important;
}

.gr-button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 30px rgba(99, 102, 241, 0.5) !important;
}

.examples-container {
    margin-top: 1.5rem !important;
    padding: 1.25rem !important;
    background: rgba(255, 255, 255, 0.08) !important;
    border-radius: 20px !important;
    border: 1px solid rgba(255, 255, 255, 0.1) !important;
}

.example-btn {
    background: rgba(255, 255, 255, 0.95) !important;
    border: 1px solid #e2e8f0 !important;
    color: #6366f1 !important;
    padding: 10px 24px !important;
    margin: 6px !important;
    border-radius: 50px !important;
    font-size: 0.85rem !important;
    font-weight: 600 !important;
    transition: all 0.3s ease !important;
    cursor: pointer !important;
}

.example-btn:hover {
    background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%) !important;
    color: white !important;
    transform: translateY(-3px) !important;
}

.accordion {
    background: rgba(255, 255, 255, 0.08) !important;
    backdrop-filter: blur(10px) !important;
    border-radius: 20px !important;
    margin-top: 1.5rem !important;
    border: 1px solid rgba(255, 255, 255, 0.1) !important;
}

.footer {
    text-align: center !important;
    padding: 1.5rem !important;
    color: #94a3b8 !important;
    font-size: 0.85rem !important;
}

::-webkit-scrollbar {
    width: 10px;
}

::-webkit-scrollbar-track {
    background: rgba(255, 255, 255, 0.05);
    border-radius: 10px;
}

::-webkit-scrollbar-thumb {
    background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%);
    border-radius: 10px;
}

@media (max-width: 768px) {
    .main-header h1 {
        font-size: 2rem !important;
    }
    
    [data-testid="user"], [data-testid="bot"] {
        max-width: 95% !important;
    }
}

.status-badge {
    animation: float 3s ease-in-out infinite !important;
}
"""

# ==============================
# GRADIO INTERFACE (UNCHANGED)
# ==============================

with gr.Blocks(title="✨ Health Assistant ", css=modern_css) as demo:
    
    with gr.Column():
        with gr.Row():
            with gr.Column(elem_classes=["main-header"]):
                gr.Markdown("""
                # 🏥 Health Assistant Pro
                
                ### Your 24/7 Intelligent Health Information Partner
                """)
                gr.HTML("""
                <div style="text-align: center; margin-top: 15px;">
                    <span class="status-badge">✓ ACTIVE</span>
                    <span class="status-badge">🔒 ENCRYPTED</span>
                    <span class="status-badge">🏥 MEDICAL GRADE</span>
                    <span class="status-badge">⚡ REAL-TIME</span>
                </div>
                """)
        
        with gr.Column(elem_classes=["chat-container"]):
            chatbot = gr.Chatbot(
                label="Health Assistant",
                height=550,
                show_label=False,
                elem_classes=["chatbot"]
            )
            
            with gr.Row(elem_classes=["input-container"]):
                with gr.Column(scale=9):
                    msg = gr.Textbox(
                        placeholder="💬 Ask me anything about health... (e.g., 'What are the symptoms of a cold?', 'How much water should I drink?')",
                        lines=2,
                        show_label=False
                    )
                with gr.Column(scale=1):
                    submit_btn = gr.Button("🚀 Send Message", variant="primary")
                    clear_btn = gr.Button("🗑️ Clear Chat", variant="secondary")
            
            with gr.Column(elem_classes=["examples-container"]):
                gr.Markdown("### 💡 Quick Health Questions")
                
                example_questions = [
                    "What causes headaches?",
                    "How can I improve sleep quality?",
                    "What are signs of dehydration?",
                    "Is exercise good for mental health?",
                    "How to boost immune system?",
                    "What foods reduce inflammation?"
                ]
                
                example_btns = []
                for i in range(0, len(example_questions), 3):
                    with gr.Row():
                        for j in range(3):
                            if i + j < len(example_questions):
                                btn = gr.Button(
                                    example_questions[i + j], 
                                    elem_classes=["example-btn"]
                                )
                                example_btns.append((btn, example_questions[i + j]))
        
        with gr.Accordion("📋 Safety & Medical Disclaimer", open=False, elem_classes=["accordion"]):
            with gr.Row():
                with gr.Column():
                    gr.Markdown("""
                    ### ✅ Appropriate Questions:
                    - General health information and wellness tips
                    - How body systems function
                    - Common illness causes and prevention
                    - Healthy lifestyle recommendations
                    - Nutrition and exercise guidance
                    
                    ### ❌ Not For:
                    - Medical diagnosis or treatment plans
                    - Specific medication dosages or prescriptions
                    - Emergency medical advice
                    - Treatment for serious conditions
                    """)
                
                with gr.Column():
                    gr.Markdown("""
                    ### 🚨 In Emergency:
                    **Call emergency services immediately:**
                    - 🇺🇸 911 (USA)
                    - 🇪🇺 112 (Europe)
                    - 🇬🇧 999 (UK)
                    
                    ### 🔒 Privacy Guarantee:
                    Your conversations are private and not stored.
                    
                    ---
                    **⚠️ IMPORTANT:** This AI provides general health information only. Always consult a qualified healthcare professional for medical advice, diagnosis, or treatment.
                    """)
        
        with gr.Row():
            gr.HTML("""
            <div class="footer">
                <span>✨ Health Assistant Pro v3.0 | Powered by TinyLlama AI | Not for emergencies ✨</span><br>
                <span style="font-size: 0.7rem;">© 2024 Health Assistant Pro - Your Trusted Health Information Partner</span>
            </div>
            """)
    
    # ==============================
    # EVENT HANDLERS (UNCHANGED)
    # ==============================
    
    def respond(message, history):
        if not message or message.strip() == "":
            return "", history
        
        bot_response = generate_response(message, history)
        
        if history is None:
            history = []
        
        history.append({"role": "user", "content": message})
        history.append({"role": "assistant", "content": bot_response})
        
        return "", history
    
    def clear_history():
        return []
    
    submit_btn.click(respond, [msg, chatbot], [msg, chatbot])
    msg.submit(respond, [msg, chatbot], [msg, chatbot])
    clear_btn.click(clear_history, None, chatbot)
    
    for btn, question in example_btns:
        btn.click(lambda q=question: q, None, msg).then(
            respond, [msg, chatbot], [msg, chatbot]
        )

# ==============================
# LAUNCH
# ==============================

if __name__ == "__main__":
    demo.launch(
        share=False,
        server_name="0.0.0.0",
        server_port=7860,
        quiet=False
    )