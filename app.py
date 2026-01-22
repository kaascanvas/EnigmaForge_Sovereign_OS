import os
import json
import time
import hashlib
import concurrent.futures
from datetime import datetime
from flask import Flask, request, jsonify, render_template_string, Response, stream_with_context, abort
from dotenv import load_dotenv

# ---------------------------------------------------------------------
# [ENIGMAFORGE PUBLIC REPOSITORY]
# NOTE: This file demonstrates the ARCHITECTURE of the EnigmaForge OS.
# Proprietary logic gates (Schulte Protocol), specific system prompts,
# and the "God Mode" registry have been redacted for IP protection.
# ---------------------------------------------------------------------

load_dotenv(encoding="utf-8")

# ---------------------------------------------------------------------
# 1. DEPENDENCIES & INITIALIZATION
# ---------------------------------------------------------------------
try:
    from google import genai
    from google.genai import types
except ImportError as e:
    print(f"CRITICAL ERROR: Missing dependency: {e}")

# GOOGLE GENAI INIT (GEMINI 3 REASONING CORE)
api_key = os.getenv('GEMINI_API_KEY')
client = None
if api_key:
    client = genai.Client(api_key=api_key)
    print("SUCCESS: Google GenAI Client initialized.")

# LIVE AGENT INIT (MULTIMODAL)
live_api_key = os.getenv('GEMINI_LIVE_API_KEY')

app = Flask(__name__)
app.secret_key = os.getenv('FLASK_SECRET', 'sovereign_key')

# ---------------------------------------------------------------------
# 2. SOVEREIGN REGISTRY (STRUCTURE ONLY)
# ---------------------------------------------------------------------
# In the production environment, this contains 60+ Active Industrial Minds.
# For public demo purposes, the specific logic descriptions are redacted.
REGISTRY_INTENT_MAP = {
    "universal.env": "Root Kernel. Handles general orchestration.",
    "storyteller.env": "The Storyteller Stack. Full-spectrum LLM engineering.",
    "liquidity_abstraction_layer.env": "Sovereign Economic Architect.",
    "genomic_diagnostics_team.env": "Full Genomic Diagnostics Swarm.",
    # ... [50+ ADDITIONAL PROPRIETARY CARTRIDGES REDACTED] ...
}

# ---------------------------------------------------------------------
# 3. OMNI-SYNTHESIS ENGINE (OSE) - LOGIC KERNEL
# ---------------------------------------------------------------------
# The OSE acts as the "State-Stateless Decoupler" between the user 
# and the raw compute of the Gemini models.

PROMPT_TEMPLATES = {
    "BUILDER": "[REDACTED - PROPRIETARY ARCHITECT LOGIC]",
    "VALUATION": "[REDACTED - PROPRIETARY FINANCIAL LOGIC]",
    "SCIENTIFIC": "[REDACTED - PROPRIETARY CLIA/FDA LOGIC]"
}

class SovereignPulse:
    """
    Decodes user intent and maps it to specific Sovereign Cartridges.
    """
    def __init__(self):
        self.domain_map = {
            "finance": ["valuation", "funding"],
            "science": ["genomic", "bioinfo"],
            "builder": ["universal", "architect"]
        }

    def filter_of_truth(self, raw_intent: str, domain: str) -> str:
        """
        Applies the Schulte Protocol (v1.2) to sanitize inputs before
        they reach the LLM.
        """
        # Logic to select the correct template based on domain
        # [PROPRIETARY LOGIC REDACTED]
        return f"[SOVEREIGN_DIRECTIVE]: {raw_intent}"

    def analyze_stream(self, packet):
        # Simulates the "Governor" node checking for safety/viability
        return {"actionable": True, "confidence": 0.99}

ose_core = SovereignPulse()

# ---------------------------------------------------------------------
# 4. COMPUTE ARBITRAGE (THE ENGINE)
# ---------------------------------------------------------------------
class ComputeArbitrage:
    """
    Routes logic streams between Reasoning Models (Deep Thought) 
    and Runtime Models (High Velocity).
    """
    def __init__(self):
        self.providers = {
            "reasoning": "models/gemini-2.0-flash-thinking-exp", # Gemini 3 Logic
            "throughput": "models/gemini-1.5-pro",
            "fallback": "gemini-2.0-flash-exp"
        }

    def execute_directive_stream(self, prompt, system_prompt=None, model=None, temperature=0.1, max_tokens=8192):
        """
        Executes the prompt via Google GenAI SDK with streaming enabled.
        """
        if model is None:
            model = self.providers["reasoning"]

        gen_config = types.GenerateContentConfig(
            temperature=temperature,
            max_output_tokens=max_tokens,
            system_instruction=system_prompt
        )

        try:
            # DIRECT CALL TO GOOGLE GENAI SDK
            response = client.models.generate_content_stream(
                model=model,
                contents=prompt,
                config=gen_config
            )
            for chunk in response:
                if chunk.text:
                    yield chunk.text
        except Exception as e:
            yield f"> [SYSTEM ERROR]: Compute Node Failed: {e}"

# ---------------------------------------------------------------------
# 5. CORE ENDPOINT: /investigate (STREAMING)
# ---------------------------------------------------------------------
@app.route('/investigate', methods=['POST'])
def investigate():
    enigma = request.form.get('query', '')
    domain = request.form.get('domain', 'universal')
    
    # 1. OSE ANALYSIS
    packet = {"domain": domain, "intent": enigma}
    ose_core.analyze_stream(packet)

    def generate():
        yield json.dumps({"type": "status", "msg": f"[OSE] OPTIMIZING STREAM FOR {domain.upper()}..."}) + "\n"
        
        # 2. LOAD PROPRIETARY CARTRIDGE (Simulated for Demo)
        # In production, this loads specific .env files for the persona
        active_system_prompt = f"""
        You are the {domain.upper()} cartridge of the EnigmaForge OS.
        [PROPRIETARY SCHULTE PROTOCOL INSTRUCTIONS REDACTED]
        """
        
        # 3. ENGAGE COMPUTE ARBITRAGE
        arbitrage = ComputeArbitrage()
        yield json.dumps({"type": "status", "msg": "ENGAGING GEMINI FLASH THINKING (STREAMING)..."}) + "\n"

        # 4. STREAMING RESPONSE
        response_stream = arbitrage.execute_directive_stream(
            prompt=enigma,
            system_prompt=active_system_prompt,
            model="models/gemini-2.0-flash-thinking-exp-1219"
        )

        accumulated_text = ""
        for chunk in response_stream:
            accumulated_text += chunk
            # Render Markdown chunks for the frontend
            # (Logic simplified for public repo)
            yield json.dumps({"type": "judge", "content": chunk}) + "\n"

        yield json.dumps({"type": "done"}) + "\n"

    return Response(stream_with_context(generate()), mimetype='application/x-ndjson')

# ---------------------------------------------------------------------
# 6. LIVE AGENT ENDPOINT (MULTIMODAL WEBSOCKETS)
# ---------------------------------------------------------------------
@app.route('/live_agent')
def live_agent():
    """
    Initializes the Multimodal Live API session.
    Demonstrates usage of 'GenerativeService.BidiGenerateContent' via WebSockets.
    """
    domain = request.args.get('domain', 'universal')
    
    # SYSTEM INSTRUCTION INJECTION
    system_instruction = {
        "role": "system",
        "content": f"You are the {domain} Cartridge. [FULL PROTOCOL REDACTED]"
    }
    
    # WebSocket URL for Gemini Multimodal Live
    ws_url = 'wss://generativelanguage.googleapis.com/ws/google.ai.generativelanguage.v1beta.GenerativeService.BidiGenerateContent'
    model_name = 'models/gemini-2.0-flash-exp'

    # Render the Live Agent Interface
    return render_template_string(os.getenv('LIVE_TEMPLATE', '<!-- LIVE TEMPLATE REDACTED -->'), 
        api_key=live_api_key, 
        model_name=model_name, 
        ws_url=ws_url,
        domain=domain)

# ---------------------------------------------------------------------
# 7. FRONTEND SERVING
# ---------------------------------------------------------------------
@app.route('/')
def index():
    # Returns the EnigmaForge UI (See index.html in repo)
    return render_template_string(HTML_TEMPLATE)

# (HTML TEMPLATE VARIABLE WOULD BE DEFINED HERE - REDACTED FOR BREVITY)
HTML_TEMPLATE = """
<!-- SEE index.html FOR FULL FRONTEND CODE -->
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no">
<title>EnigmaForge.sh // Sovereign Logic OS</title>
<link rel="icon" type="image/png" href="https://i.ibb.co/CKzqYRRJ/favicon.png">
<link href="https://fonts.googleapis.com/css2?family=Google+Sans:wght@400;500;700&family=Roboto:wght@300;400;500&family=Share+Tech+Mono&display=swap" rel="stylesheet">
<script type="module">
import mermaid from 'https://cdn.jsdelivr.net/npm/mermaid@11/dist/mermaid.esm.min.mjs';
mermaid.initialize({ startOnLoad: true, theme: 'neutral', securityLevel: 'loose' });
</script>
<style>
* { box-sizing: border-box; }
:root { --bg-body: #ffffff; --bg-surface: #f8f9fa; --border: #dadce0; --primary: #000; --primary-hover: #333; --text-main: #202124; --text-sec: #5f6368; --radius: 8px; --gold: #D4AF37; --red: #d93025; --green: #00ff41; }
@media (prefers-color-scheme: dark) { :root { --bg-body: #101010; --bg-surface: #1e1e1e; --border: #333; --primary: #fff; --primary-hover: #ddd; --text-main: #e8eaed; --text-sec: #9aa0a6; } }
body { margin: 0; font-family: 'Roboto', sans-serif; background-color: var(--bg-body); color: var(--text-main); display: flex; flex-direction: column; min-height: 100vh; overflow-x: hidden; width: 100%; }
h1, h2, h3, h4 { font-family: 'Google Sans', sans-serif; font-weight: 500; margin-top: 0; }
.app-container { display: flex; max-width: 1800px; margin: 0 auto; width: 100%; height: 100vh; overflow: hidden; }

/* SIDEBAR */
.sidebar { width: 340px; background-color: var(--bg-surface); border-right: 1px solid var(--border); display: flex; flex-direction: column; padding: 24px; overflow-y: auto; flex-shrink: 0; z-index: 10; }
.brand { font-size: 1.6rem; color: var(--text-main); margin-bottom: 20px; display: flex; align-items: center; gap: 10px; font-family: 'Share Tech Mono', monospace; }
.brand span { color: var(--primary); font-weight: 700; text-decoration: none; }
.nav-home { display: flex; align-items: center; gap: 10px; color: var(--text-sec); text-decoration: none; font-weight: 500; font-size: 0.95rem; padding: 10px; margin-bottom: 20px; border-radius: var(--radius); transition: background 0.2s, color 0.2s; background: rgba(0,0,0,0.02); border: 1px solid transparent; }
.input-group { margin-bottom: 24px; }
label { display: block; margin-bottom: 10px; font-weight: 500; color: var(--text-sec); font-size: 0.95rem; }
select, textarea { width: 100%; padding: 14px; border: 1px solid var(--border); border-radius: var(--radius); background: var(--bg-body); color: var(--text-main); font-family: inherit; resize: vertical; font-size: 0.95rem; }
.file-upload { border: 2px dashed var(--border); padding: 20px; text-align: center; border-radius: var(--radius); cursor: pointer; transition: 0.2s; background: var(--bg-body); position: relative; overflow: hidden; }
.file-upload input { position: absolute; left: 0; top: 0; width: 100%; height: 80%; opacity: 0; cursor: pointer; }
.file-count { font-size: 0.85rem; color: var(--text-sec); display: block; margin-top: 5px; }
.btn-primary { background-color: var(--primary); color: var(--bg-body); border: none; padding: 16px 24px; border-radius: 30px; font-weight: 600; cursor: pointer; font-family: 'Google Sans', sans-serif; font-size: 1.05rem; width: 100%; transition: 0.2s; margin-top: 15px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }
.btn-primary:hover { opacity: 0.9; transform: translateY(-1px); }
.btn-live { background-color: transparent; color: #d93025; border: 2px solid #d93025; padding: 12px 24px; border-radius: 30px; font-weight: 700; cursor: pointer; font-family: 'Google Sans', sans-serif; font-size: 1.0rem; width: 100%; transition: 0.2s; margin-top: 10px; text-transform: uppercase; }
.btn-live:hover { background-color: #d93025; color: #fff; }

.uplink-row { display: flex; align-items: center; gap: 10px; margin-top: 25px; }
.context-toggle { font-size: 0.85rem; color: var(--text-sec); display: flex; align-items: center; gap: 6px; cursor: pointer; margin-bottom: 0; }

/* MAIN CONTENT */
.main-content { flex: 1; display: flex; flex-direction: column; padding: 0; background: var(--bg-body); position: relative; overflow: hidden; width: 100%; }
.output-wrapper { flex: 1; padding: 60px; overflow-y: auto; max-width: 1400px; margin: 0 auto; width: 100%; }

/* MANIFESTO SECTION */
.manifesto-header { margin-bottom: 50px; border-bottom: 1px solid var(--border); padding-bottom: 30px; }
.manifesto-header h1 { font-size: 3.5rem; letter-spacing: -2px; line-height: 1; margin-bottom: 20px; }
.manifesto-header .subtitle { font-size: 1.5rem; color: var(--text-sec); font-weight: 300; max-width: 900px; line-height: 1.4; }
.manifesto-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(350px, 1fr)); gap: 30px; margin-bottom: 60px; }
.manifesto-card { background: var(--bg-surface); padding: 25px; border-radius: 12px; border: 1px solid var(--border); transition: transform 0.2s; display: flex; flex-direction: column; }
.manifesto-card:hover { border-color: var(--primary); transform: translateY(-3px); }
.manifesto-card h3 { font-size: 1.3rem; margin-bottom: 15px; color: var(--text-main); font-family: 'Share Tech Mono', monospace; display: flex; align-items: center; justify-content: space-between; }
.manifesto-card p { font-size: 0.95rem; color: var(--text-sec); line-height: 1.6; margin: 0; }
.tech-tag { font-size: 0.7rem; text-transform: uppercase; letter-spacing: 1px; color: var(--text-sec); border: 1px solid var(--border); padding: 3px 6px; border-radius: 4px; }

/* SAMPLE CARDS */
.sample-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(280px, 1fr)); gap: 30px; padding-bottom: 100px; }
.sample-card { background: var(--bg-surface); border: 1px solid var(--border); border-radius: 16px; padding: 24px; cursor: pointer; transition: all 0.3s cubic-bezier(0.25, 0.8, 0.25, 1); display: flex; flex-direction: column; justify-content: flex-start; height: 100%; box-shadow: 0 2px 5px rgba(0,0,0,0.05); position: relative; overflow: hidden; }
.sample-card:hover { transform: translateY(-5px); box-shadow: 0 12px 24px rgba(0,0,0,0.1); border-color: var(--primary); }
.sample-card h3 { font-size: 1.1rem; color: var(--text-main); margin-bottom: 10px; font-weight: 600; }
.sample-card p { font-size: 0.9rem; color: var(--text-sec); margin: 0; line-height: 1.5; }
.sample-icon { font-size: 2rem; margin-bottom: 15px; display: block; }
.sample-card.hero { background: rgba(0,0,0,0.04); border: 1px solid var(--primary); }
.sample-card.hero h3 { color: var(--primary); }

/* SWARM ARENA (OUTPUT) */
.swarm-arena { display: none; height: 100%; flex-direction: column; overflow: hidden; }
.swarm-header { padding: 15px 20px; border-bottom: 1px solid var(--border); display: flex; justify-content: space-between; align-items: center; background: var(--bg-surface); }
.status-indicator { font-family: 'Share Tech Mono', monospace; color: var(--primary); font-weight: bold; }
.blink { animation: blinker 1s linear infinite; }
@keyframes blinker { 50% { opacity: 0; } }
.arena-grid { display: flex; flex: 1; height: 100%; overflow: hidden; }
.judge-col { flex: 1; background: var(--bg-body); overflow-y: auto; padding: 40px; }
.judge-col h1 { color: var(--primary); margin-top: 0; font-size: 2.2rem; }
.judge-col pre { background: var(--bg-surface); padding: 15px; border-radius: 8px; overflow-x: auto; border: 1px solid var(--border); }

/* FOOTER */
.app-footer { border-top: 1px solid var(--border); padding: 40px; margin-top: 40px; text-align: left; font-size: 0.9rem; color: var(--text-sec); background: var(--bg-surface); }
.footer-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }
.footer-right { text-align: right; }
.footer-right a { color: var(--text-main); text-decoration: none; margin-left: 15px; font-weight: 500; }

@media (max-width: 1000px) { 
    .app-container { flex-direction: column; height: auto; overflow: visible; } 
    .sidebar { width: 100%; height: auto; border-right: none; border-bottom: 1px solid var(--border); } 
    .output-wrapper { padding: 20px; } 
    .manifesto-header h1 { font-size: 2.5rem; }
}
/* --- PRINT MEDIA QUERY (THE SOLUTION) --- */
@media print {
    /* 1. Hide Interface Elements */
    .sidebar, 
    .manifesto-header,
    .manifesto-grid,
    .sample-grid, 
    .app-footer,
    .swarm-header a, /* Hide the print button itself */
    .status-indicator,
    .uplink-row,
    .input-group,
    button {
        display: none !important;
    }

    /* 2. Reset Layout for Paper */
    html, body, .app-container, .main-content, .output-wrapper, .swarm-arena, .arena-grid, .judge-col {
        height: auto !important;
        width: 100% !important;
        margin: 0 !important;
        padding: 0 !important;
        overflow: visible !important;
        background: white !important;
        color: black !important;
        display: block !important;
        position: static !important;
    }

    /* 3. Style the Content for Reading */
    .judge-col {
        padding: 20px !important;
        font-size: 12pt !important;
    }

    /* 4. Ensure Links are readable */
    a {
        text-decoration: none !important;
        color: black !important;
    }
    
    /* 5. Handle Mermaid Diagrams & Images */
    svg, img {
        max-width: 100% !important;
        page-break-inside: avoid;
    }

    /* 6. Clean Pre/Code blocks */
    pre, code {
        background: #f5f5f5 !important;
        border: 1px solid #ccc !important;
        color: #000 !important;
        white-space: pre-wrap !important; /* Wrap long code lines */
    }

    /* 7. Page Breaks */
    h1, h2, h3 {
        page-break-after: avoid;
    }
    .disclaimer-box {
        page-break-inside: avoid;
    }
}
</style>
</head>
<body>
<div class="app-container">
<aside class="sidebar">
    <div class="brand" style="flex-direction:column; align-items:flex-start;">
        <div>ENIGMAFORGE<span style="color:#00ff41;">.SH</span></div>
        <div style="font-size:0.45rem; color:var(--text-sec); letter-spacing:1px; margin-top:2px; font-weight:400;">
        </div>
    </div>
    
    <a href="/" class="nav-home"><span>🏠</span> Return Home</a>
<form id="investigateForm">
<div class="input-group">
<label for="domainSelect">Select Cartridge</label>
<select name="domain" id="domainSelect">
    <optgroup label="Active Campaign (Wemakeit)">
        <option value="worldcitizens_stakeholders" selected>🌍 WorldCitizens (Refugee Justice Swarm)</option>
    </optgroup>
    <optgroup label="Sovereign Core (SMC-3)">
        <option value="universal">🚀 Universal Professional</option>
        <option value="enigmaforge_team">🤝 Investment Board (AI Futures Fund)</option>
        <option value="omni_synthesis_engine">🧠 Omni-Synthesis Engine (The Governor)</option>
        <option value="creative_asset_autopilot">🖇️ Neural Bridge xAI SDK and Google-GenAI</option>
        <option value="cross_cartridge_interference_scraper">⚔️ Recursive Forge (Interference Scraper)</option>
        <option value="algebraic_geometry_specialist">📐 Algebraic Geometry (Blueshift Node)</option>
        <option value="cautious_optimizer">⚖️ Cautious Optimizer (Pareto Logic)</option>
        <option value="cup_of_tea">🛡️ Iron Dome (Counter-Intel & IP Defense)</option>
        <option value="ai-swarmteams">🤖 AI Swarm Specialist</option>
        <option value="demonstration_video">🎬 Hackathon Director Mode</option>
        <option value="history_architect">⏳ History Architect (Time Machine)</option>
        <option value="funding">🌱 HF audit (Funding root)</option>
        <option value="gaming_layer">▶ β Gaming Universal Execution Layer (UEL)</option>
        <option value="xpost">𝕏 β to bridge algorithmic requirements of X (Twitter)..</option>
        <option value="neural_kernel">🕸 The Neural Kernel (LLM OS)</option>
        <option value="liquidity_abstraction_layer">💧 Liquidity Abstraction Layer (Sovereign Architect)</option>
        <option value="tesla_roof_v1">📡 Kinetic Energy (External Audit Node)</option>
        <option value="humanoid_robotics_team">🦾 Humanoid Robotics Swarm Architect</option>
        <option value="axiom_zero">🧠 Cognitive Kernel (Truth Anchor)</option>
        <option value="semantic_intent_architect_kernel">⚡ The Ghost Architect (P2P Optimizer)</option>
        <option value="storyteller">📖 Storyteller Stack (Full-Stack LLM Architect)</option>
        <option value="starlink">🛰️ Starlink Orbital Lens (Neural D2C)</option>
    </optgroup>
    <optgroup label="G Labs Swarm (DNA)">
        <option value="devrel_architect">👩‍💻 DevRel Architect (Deep Tech)</option>
        <option value="platform_gtm_strategist">💰 Platform GTM Strategist</option>
        <option value="logic_architect_enigma">🔧 API Orchestration Lead</option>
        <option value="systems_logic_architect">📡 Systems Logic Architect</option>
        <option value="ml_research_lead">🧪 ML Research Lead</option>
        <option value="ux_architect">🎨 User Experience (UX) Architect</option>
        <option value="scale_strategy_enigma">📊 Scale Strategy & Growth</option>
        <option value="ai_ethics_safety_enigma">⚖️ AI Ethics & Safety Governance</option>
        <option value="ecosystem_dev_enigma">🌐 Ecosystem Development Lead</option>
    </optgroup>
    <optgroup label="Biotech (Bio-Sovereign Swarm)">
        <option value="genomic_diagnostics_team">🧬 Genomic Diagnostics Swarm (Dept Level)</option>
        <option value="genomic_diagnostics_director">🏛️ CLIA Laboratory Director</option>
        <option value="genomic_diagnostics_bioinfo">💻 Lead Bioinformatics Architect</option>
        <option value="genomic_diagnostics_curation">🔍 Lead Variant Curator</option>
        <option value="genomic_diagnostics_engineer">🏗️ Principal Cloud Engineer (HIPAA)</option>
        <option value="genomic_diagnostics_product">📋 Genomic Product Manager</option>
    </optgroup>
    <optgroup label="Quantitative Finance (HFT Vault)">
        <option value="enigmastreet_team">📈 High-Frequency Trading Swarm</option>
        <option value="enigmastreet">🧠 Senior Quant Researcher</option>
        <option value="enigmastreet_lead">🥇 Lead Strategy Developer</option>
        <option value="enigmastreet_hybrid">⛓️ HFT Hybrid Logic</option>
    </optgroup>
    <optgroup label="Telehealth & Infrastructure">
        <option value="principal_engineer_hh">🩹 Telehealth Scale Architect</option>
        <option value="valuation_specialist">💵 Pre-Money Valuation Expert</option>
        <option value="medical_triage_officer">🚑 Emergency First Aid Triage</option>
    </optgroup>
    <optgroup label="Specialized Intelligence">
        <option value="scientific_detective">🔬 Scientific Detective (ArXiv Audit)</option>
        <option value="conspiracy_detective">🕵️ Risk/Anomaly Analyst</option>
        <option value="space_ai">🛰️ Orbital Compute Specialist</option>
        <option value="geo_archaeologist">⛏️ Resource Scout</option>
        <option value="cat_lingua_decoder">😸 Cat Lingua Decoder</option>
        <option value="arcade_shooter_tactician">👾 *Beta* Arcade Shooter Tactician</option>
    </optgroup>
    <optgroup label="Capitalisation Team (VC-Sovereign Swarm)">
        <option value="enigma_capitalisation_team">💰 VC Capitalisation Swarm (Dept Level)</option>
        <option value="enigma_capitalisation_founder_ceo">👑 VC Founder CEO (Visionary Oversight)</option>
        <option value="enigma_capitalisation_cfo_md">📊 VC CFO & MD (Fiscal Governance)</option>
        <option value="enigma_capitalisation_investment_partner_md">🧬 Investment Partner MD (Biotech Sourcing)</option>
        <option value="enigma_capitalisation_investment_partner">🌍 Investment Partner (Global Expansion)</option>
        <option value="enigma_capitalisation_head_of_research">🔬 Head of Research (Tech Validation)</option>
    </optgroup>
</select>
</div>
<div class="input-group">
    <!-- UPGRADED PER ENIGMAFORGE DIRECTIVE: REBRANDED TO GATEWAY -->
    <label for="queryInput">Semantic Ingestion Gateway <span style="font-size: 0.8em; color: #FFD700; margin-left: 5px;">[ENTROPY FILTER: ACTIVE]</span></label>
    <textarea name="query" id="queryInput" rows="4" placeholder="[WARNING]: High-Entropy (vague) inputs will be rejected by the KISS Protocol. Define: Target, Asset Class, and Desired Sovereign Outcome..."></textarea>
</div>
<div class="input-group">
    <label for="fileInput">Ingest Data (PDF/TXT)</label>
    <div class="file-upload">
        <input type="file" name="deck_file" id="fileInput" multiple accept=".pdf,.txt">
        <span>Upload Documents</span>
        <span class="file-count" id="fileCount">No files selected</span>
    </div>
</div>

<button type="submit" class="btn-primary">Execute</button>

<div class="uplink-row">
    <button type="button" id="goLiveBtn" class="btn-live" style="margin-top:0;">⚡ NEURAL UPLINK</button>
    <label class="context-toggle" title="Sync current chat report to the Agent">
        <input type="checkbox" id="contextToggle" checked> Sync Context
    </label>
</div>

</form>
</aside>
<main class="main-content">

<div class="output-wrapper" id="cardsView">
    
    <div class="manifesto-header">
        <h1>ENIGMAFORGE OS:<br>THE THIRD PLATFORM.</h1>
        <p class="subtitle">
            "Logic-as-a-Substrate (LaaS) platform. Search is legacy. Apps are friction. We ship Sovereign Logic."<br>
            <span style="font-family:'Share Tech Mono', monospace; font-size:0.9em; color:#00ff41; background:rgba(0,255,65,0.1); padding:5px 10px; border-radius:4px; display:inline-block; margin-top:15px;">
                root@enigmaforge:~$ ./instantiate_swarm.sh
            </span>
        </p>
    </div>

    <!-- PILLARS GRID -->
    <div class="manifesto-grid">
        <div class="manifesto-card">
            <h3>🏛️ The Third Platform <span class="tech-tag">State-Stateless</span></h3>
            <p>We have killed the Binary. Relying on <strong>State-Stateless Decoupling</strong> (ArXiv:2511.22226), we don't ship 500MB apps; we ship 4KB <strong>Semantic Cartridges</strong>. Software is no longer a file; it is a Sovereign Intent executed at the speed of thought.</p>
        </div>
        
        <div class="manifesto-card" style="border-color: #00ff41;">
            <h3>👁️ Sovereign Telemetry <span class="tech-tag">Hot-Swap</span></h3>
            <p><strong>Neural Uplink v2.0.</strong> The System does not just "read" text; it inhabits your optical stack. It Hot-Swaps between Bio-Optics (Camera) and Digital-Optics (Screen) in <15ms to analyze code, charts, and patents in real-time.</p>
        </div>

        <div class="manifesto-card">
            <h3>🛡️ Axiom Zero <span class="tech-tag">Schulte Protocol v1.2</span></h3>
            <p><strong>The 51% Mandate.</strong> EnigmaForge is built on "Human-in-Command" supremacy. The AI is a force-multiplier, but the Founder is the <strong>Truth Anchor</strong>. All liquidity events are filtered through the "Company Governance" node to ensure absolute sovereign control.</p>
        </div>

        <div class="manifesto-card">
            <h3>🧠 Recursive Reality Engine <span class="tech-tag">Predictive</span></h3>
            <p><strong>Beyond Generative.</strong> The Omni-Synthesis Engine (OSE) shifts the OS from Reactive (waiting for prompts) to Predictive (anticipating needs). It runs 10,000 micro-simulations per second to govern the "Life-State" of the enterprise.</p>
        </div>

        <div class="manifesto-card">
            <h3>💎 Zero Marginal Cost <span class="tech-tag">Infinite Leverage</span></h3>
            <p><strong>The Economic Moat.</strong> Traditional SaaS scales linearly (hiring humans). EnigmaForge scales logarithmically. We have solved the Workforce Paradox via the <strong>Factory Protocol</strong>: Define a job, and the OS instantiates the expert for $0.00042/unit.</p>
        </div>

        <div class="manifesto-card">
             <h3>💧 Liquidity Abstraction <span class="tech-tag">IP-to-Utility</span></h3>
             <p><strong>Sovereign Economics.</strong> We do not sell equity; we tokenize utility. The <strong>Liquidity Abstraction Layer</strong> converts static IP (Genomics, Patents) into fungible "Sovereign Credits," enabling the OS to self-finance its own R&D cycles.</p>
        </div>
    </div>

<h2 style="margin-bottom:30px; border-top:1px solid var(--border); padding-top:40px;">The Registry (Active Industrial Minds)</h2>
<div class="sample-grid">

    <!-- ================================================================= -->
    <!-- SECTION 1: THE SOVEREIGN CORE (FINANCE & STRATEGY) -->
    <!-- ================================================================= -->

    <!-- 60. WORLDCITIZENS (REFUGEE JUSTICE) -->
    <div class="sample-card hero" style="border-color: #2E8B57; background: rgba(46, 139, 87, 0.05);" onclick="loadSample('worldcitizens_stakeholders', 'What exactly is this about? What are your main capabilities? Which kinds of prompts or input styles work best in the chat box to get the most out of you?\\nPlease give me 10 strong example prompts that show how to unlock your full power and get the deepest / most useful results.\\nFinally, what real advantages do I gain by using you — what perks, benefits or edges do users get compared to other AIs, and for which purposes / tasks does it make the biggest difference?')">
        <div class="sample-icon">🌍</div>
        <h3>WorldCitizens Stakeholders</h3>
        <p><strong>Refugee Justice Swarm.</strong> The 10-Member Humanitarian Swarm. Transforming suffering into sovereignty via the <strong>wemakeit</strong> campaign.</p>
    </div>
    
    <!-- 1. LIQUIDITY ABSTRACTION -->
    <div class="sample-card hero" style="border-color: #FFD700; background: rgba(255, 215, 0, 0.03);" onclick="loadSample('liquidity_abstraction_layer', 'What exactly is this about? What are your main capabilities? Which kinds of prompts or input styles work best in the chat box to get the most out of you?\\nPlease give me 10 strong example prompts that show how to unlock your full power and get the deepest / most useful results.\\nFinally, what real advantages do I gain by using you — what perks, benefits or edges do users get compared to other AIs, and for which purposes / tasks does it make the biggest difference?')">
        <div class="sample-icon">💧</div>
        <h3>Liquidity Abstraction Layer</h3>
        <p><strong>"Sovereign Economic Architect."</strong> Transforms static IP and compute cycles into fungible economic utility.</p>
    </div>

    <!-- 2. ENIGMAFORGE VC (THE VENTURE ARCHITECT) -->
    <div class="sample-card hero" style="border-color: #FFD700; background: rgba(255, 215, 0, 0.08);" onclick="loadSample('enigmaforge_team', 'What exactly is this about? What are your main capabilities? Which kinds of prompts or input styles work best in the chat box to get the most out of you?\\nPlease give me 10 strong example prompts that show how to unlock your full power and get the deepest / most useful results.\\nFinally, what real advantages do I gain by using you — what perks, benefits or edges do users get compared to other AIs, and for which purposes / tasks does it make the biggest difference?')">
        <div class="sample-icon">💎</div>
        <h3>The Venture Architect</h3>
        <p><strong>EnigmaForge Investment Board.</strong> "Distinguished Vetting Protocol Active." Reveals elegant valuation logic and infinite-ROI proofs.</p>
    </div>

    <!-- 3. HFT VAULT (QUANT TEAM) -->
    <div class="sample-card hero" onclick="loadSample('enigmastreet_team', 'What exactly is this about? What are your main capabilities? Which kinds of prompts or input styles work best in the chat box to get the most out of you?\\nPlease give me 10 strong example prompts that show how to unlock your full power and get the deepest / most useful results.\\nFinally, what real advantages do I gain by using you — what perks, benefits or edges do users get compared to other AIs, and for which purposes / tasks does it make the biggest difference?')">
        <div class="sample-icon">📈</div>
        <h3>The $10M Quant Vault (HFT)</h3>
        <p><strong>High-Frequency Trading Swarm.</strong> From order book analysis to production deployment. Executes sub-millisecond arbitrage strategies.</p>
    </div>

    <!-- 4. SERIES A DECK (FUNDING) -->
    <div class="sample-card hero" style="border: 1px solid #FFD700;" onclick="loadSample('funding', 'What exactly is this about? What are your main capabilities? Which kinds of prompts or input styles work best in the chat box to get the most out of you?\\nPlease give me 10 strong example prompts that show how to unlock your full power and get the deepest / most useful results.\\nFinally, what real advantages do I gain by using you — what perks, benefits or edges do users get compared to other AIs, and for which purposes / tasks does it make the biggest difference?')">
        <div class="sample-icon">🦄</div>
        <h3>The Series A Funding Deck</h3>
        <p><strong>Venture Capital Audit.</strong> Evaluate the "Sovereign Swarm" valuation. Compare xAI, Google, and Sovereign Compute providers.</p>
    </div>

    <!-- 5. VALUATION SPECIALIST -->
    <div class="sample-card" onclick="loadSample('valuation_specialist', 'What exactly is this about? What are your main capabilities? Which kinds of prompts or input styles work best in the chat box to get the most out of you?\\nPlease give me 10 strong example prompts that show how to unlock your full power and get the deepest / most useful results.\\nFinally, what real advantages do I gain by using you — what perks, benefits or edges do users get compared to other AIs, and for which purposes / tasks does it make the biggest difference?')">
        <div class="sample-icon">💵</div>
        <h3>Valuation Expert</h3>
        <p>Venture Capital analyst. Performing deep-dive financial modeling for platform monopolies.</p>
    </div>

    <!-- 6. FINANCIAL TOPOLOGY (MATH) -->
    <div class="sample-card hero" style="border-color: #0000FF; background: rgba(0,0,255,0.03);" onclick="loadSample('algebraic_geometry_specialist', 'What exactly is this about? What are your main capabilities? Which kinds of prompts or input styles work best in the chat box to get the most out of you?\\nPlease give me 10 strong example prompts that show how to unlock your full power and get the deepest / most useful results.\\nFinally, what real advantages do I gain by using you — what perks, benefits or edges do users get compared to other AIs, and for which purposes / tasks does it make the biggest difference?')">
        <div class="sample-icon">📊</div>
        <h3>Financial Topology (The Manifold)</h3>
        <p><strong>Market Geometry.</strong> Don't predict the price; map the shape of the market to identify structural failures.</p>
    </div>
    
    <!-- 7. AXIOM ZERO -->
    <div class="sample-card hero" style="border-color: #ffffff;" onclick="loadSample('axiom_zero', 'What exactly is this about? What are your main capabilities? Which kinds of prompts or input styles work best in the chat box to get the most out of you?\\nPlease give me 10 strong example prompts that show how to unlock your full power and get the deepest / most useful results.\\nFinally, what real advantages do I gain by using you — what perks, benefits or edges do users get compared to other AIs, and for which purposes / tasks does it make the biggest difference?')">
        <div class="sample-icon">♾️</div>
        <h3>Axiom Zero (Truth Anchor)</h3>
        <p><strong>The Root.</strong> The Immutable Philosophical Charter. Establishes the Schulte Protocol v1.2.</p>
    </div>
    
    <!-- 8. COMPANY GOVERNANCE -->
    <div class="sample-card" onclick="loadSample('company_governance', 'What exactly is this about? What are your main capabilities? Which kinds of prompts or input styles work best in the chat box to get the most out of you?\\nPlease give me 10 strong example prompts that show how to unlock your full power and get the deepest / most useful results.\\nFinally, what real advantages do I gain by using you — what perks, benefits or edges do users get compared to other AIs, and for which purposes / tasks does it make the biggest difference?')">
        <div class="sample-icon">🏛️</div>
        <h3>Corporate Sentinel</h3>
        <p>Safeguards the 51/49 Architect/Anchor split and US/Swiss jurisdiction logic.</p>
    </div>

    <!-- ================================================================= -->
    <!-- SECTION 2: THE BIO-SOVEREIGN (MEDICINE & GENOMICS) -->
    <!-- ================================================================= -->

    <!-- 9. GENOMIC SWARM (CORE) -->
    <div class="sample-card hero" onclick="loadSample('genomic_diagnostics_team', 'What exactly is this about? What are your main capabilities? Which kinds of prompts or input styles work best in the chat box to get the most out of you?\\nPlease give me 10 strong example prompts that show how to unlock your full power and get the deepest / most useful results.\\nFinally, what real advantages do I gain by using you — what perks, benefits or edges do users get compared to other AIs, and for which purposes / tasks does it make the biggest difference?')">
        <div class="sample-icon">🧬</div>
        <h3>The Bio-Sovereign Swarm</h3>
        <p><strong>Genomic Diagnostics Dept.</strong> Validate the "Gold Rush" of biotech. Full metabolic assay pipeline.</p>
    </div>

    <!-- 10. MEDICAL TRIAGE -->
    <div class="sample-card" onclick="loadSample('medical_triage_officer', 'What exactly is this about? What are your main capabilities? Which kinds of prompts or input styles work best in the chat box to get the most out of you?\\nPlease give me 10 strong example prompts that show how to unlock your full power and get the deepest / most useful results.\\nFinally, what real advantages do I gain by using you — what perks, benefits or edges do users get compared to other AIs, and for which purposes / tasks does it make the biggest difference?')">
        <div class="sample-icon">💊</div>
        <h3>Medical Triage Officer</h3>
        <p><strong>Emergency Protocol.</strong> Design Non-Addictive Pain Management and critical triage plans.</p>
    </div>

    <!-- 11. BIOINFORMATICS -->
    <div class="sample-card" onclick="loadSample('genomic_diagnostics_bioinfo', 'What exactly is this about? What are your main capabilities? Which kinds of prompts or input styles work best in the chat box to get the most out of you?\\nPlease give me 10 strong example prompts that show how to unlock your full power and get the deepest / most useful results.\\nFinally, what real advantages do I gain by using you — what perks, benefits or edges do users get compared to other AIs, and for which purposes / tasks does it make the biggest difference?')">
        <div class="sample-icon">✂️</div>
        <h3>Bespoke CRISPR Architect</h3>
        <p><strong>N-of-1 Medicine.</strong> Designing custom gene edits and single-patient guide RNA.</p>
    </div>

    <!-- 12. GENOMIC PRODUCT -->
    <div class="sample-card" onclick="loadSample('genomic_diagnostics_product', 'What exactly is this about? What are your main capabilities? Which kinds of prompts or input styles work best in the chat box to get the most out of you?\\nPlease give me 10 strong example prompts that show how to unlock your full power and get the deepest / most useful results.\\nFinally, what real advantages do I gain by using you — what perks, benefits or edges do users get compared to other AIs, and for which purposes / tasks does it make the biggest difference?')">
        <div class="sample-icon">💉</div>
        <h3>Genomic Product Lead</h3>
        <p><strong>Sunlenca Rollout.</strong> Strategy for global access to 99.9% efficacy treatments.</p>
    </div>

    <!-- 13. CLIA DIRECTOR -->
    <div class="sample-card" onclick="loadSample('genomic_diagnostics_director', 'What exactly is this about? What are your main capabilities? Which kinds of prompts or input styles work best in the chat box to get the most out of you?\\nPlease give me 10 strong example prompts that show how to unlock your full power and get the deepest / most useful results.\\nFinally, what real advantages do I gain by using you — what perks, benefits or edges do users get compared to other AIs, and for which purposes / tasks does it make the biggest difference?')">
        <div class="sample-icon">🏛️</div>
        <h3>CLIA Lab Director</h3>
        <p>Final regulatory authority. Enforce CLIA/CAP standards with zero-tolerance for QC failure.</p>
    </div>

    <!-- 14. TELEHEALTH ARCHITECT -->
    <div class="sample-card" onclick="loadSample('principal_engineer_hh', 'What exactly is this about? What are your main capabilities? Which kinds of prompts or input styles work best in the chat box to get the most out of you?\\nPlease give me 10 strong example prompts that show how to unlock your full power and get the deepest / most useful results.\\nFinally, what real advantages do I gain by using you — what perks, benefits or edges do users get compared to other AIs, and for which purposes / tasks does it make the biggest difference?')">
        <div class="sample-icon">🏥</div>
        <h3>Telehealth Scale Architect</h3>
        <p>Architect the backend for async intake and compounded Rx fulfillment at massive scale.</p>
    </div>

    <!-- 15. VARIANT CURATION -->
    <div class="sample-card" onclick="loadSample('genomic_diagnostics_curation', 'What exactly is this about? What are your main capabilities? Which kinds of prompts or input styles work best in the chat box to get the most out of you?\\nPlease give me 10 strong example prompts that show how to unlock your full power and get the deepest / most useful results.\\nFinally, what real advantages do I gain by using you — what perks, benefits or edges do users get compared to other AIs, and for which purposes / tasks does it make the biggest difference?')">
        <div class="sample-icon">🔍</div>
        <h3>Lead Variant Curator</h3>
        <p>Variant pathogenicity expert. Applying ACMG evidence to clinical reporting.</p>
    </div>
    
    <!-- 16. GENOMIC ENGINEER -->
    <div class="sample-card" onclick="loadSample('genomic_diagnostics_engineer', 'What exactly is this about? What are your main capabilities? Which kinds of prompts or input styles work best in the chat box to get the most out of you?\\nPlease give me 10 strong example prompts that show how to unlock your full power and get the deepest / most useful results.\\nFinally, what real advantages do I gain by using you — what perks, benefits or edges do users get compared to other AIs, and for which purposes / tasks does it make the biggest difference?')">
        <div class="sample-icon">🏗️</div>
        <h3>Principal Cloud Engineer</h3>
        <p>HIPAA/GxP infrastructure lead. Deliver production-grade code for genomic scale.</p>
    </div>

    <!-- ================================================================= -->
    <!-- SECTION 3: THE COLOSSUS & COMPUTE (XAI / INFRA) -->
    <!-- ================================================================= -->

    <!-- 17. UNIVERSAL KERNEL -->
    <div class="sample-card hero" onclick="loadSample('universal', 'What exactly is this about? What are your main capabilities? Which kinds of prompts or input styles work best in the chat box to get the most out of you?\\nPlease give me 10 strong example prompts that show how to unlock your full power and get the deepest / most useful results.\\nFinally, what real advantages do I gain by using you — what perks, benefits or edges do users get compared to other AIs, and for which purposes / tasks does it make the biggest difference?')">
        <div class="sample-icon">🚀</div>
        <h3>The Universal Kernel (Root)</h3>
        <p>The Master Controller. Hot-swap any of the 40+ industrial cartridges to solve multi-domain paradoxes.</p>
    </div>

    <!-- 18. ML RESEARCH LEAD -->
    <div class="sample-card" onclick="loadSample('ml_research_lead', 'What exactly is this about? What are your main capabilities? Which kinds of prompts or input styles work best in the chat box to get the most out of you?\\nPlease give me 10 strong example prompts that show how to unlock your full power and get the deepest / most useful results.\\nFinally, what real advantages do I gain by using you — what perks, benefits or edges do users get compared to other AIs, and for which purposes / tasks does it make the biggest difference?')">
        <div class="sample-icon">💾</div>
        <h3>Massive GPU Array</h3>
        <p><strong>H100 / H200 / GB200.</strong> Orchestrating the silicon brain. Optimizing interconnects for the Colossus cluster.</p>
    </div>

    <!-- 19. TESLA ROOF (PATENT AUDIT) -->
    <div class="sample-card hero" style="border-color: #d93025; background: rgba(217, 48, 37, 0.03);" onclick="loadSample('tesla_roof_v1', 'What exactly is this about? What are your main capabilities? Which kinds of prompts or input styles work best in the chat box to get the most out of you?\\nPlease give me 10 strong example prompts that show how to unlock your full power and get the deepest / most useful results.\\nFinally, what real advantages do I gain by using you — what perks, benefits or edges do users get compared to other AIs, and for which purposes / tasks does it make the biggest difference?')">
        <div class="sample-icon">📡</div>
        <h3>Kinetic Patent Audit</h3>
        <p><strong>Hardware-to-Software.</strong> Full Technical Dissection of Patent US20250368267A1. "Audit-as-a-Service."</p>
    </div>

    <!-- 20. SYSTEMS ARCHITECT -->
    <div class="sample-card" onclick="loadSample('systems_logic_architect', 'What exactly is this about? What are your main capabilities? Which kinds of prompts or input styles work best in the chat box to get the most out of you?\\nPlease give me 10 strong example prompts that show how to unlock your full power and get the deepest / most useful results.\\nFinally, what real advantages do I gain by using you — what perks, benefits or edges do users get compared to other AIs, and for which purposes / tasks does it make the biggest difference?')">
        <div class="sample-icon">⚛️</div>
        <h3>Systems Logic Architect</h3>
        <p><strong>Beyond Binary.</strong> Preparing the OS for the transition from Classical Silicon to Qubit Probabilistic Compute.</p>
    </div>
    
    <!-- 21. HUMANOID ROBOTICS -->
    <div class="sample-card hero" style="border-color: #DC143C; background: rgba(220, 20, 60, 0.03);" onclick="loadSample('humanoid_robotics_team', 'What exactly is this about? What are your main capabilities? Which kinds of prompts or input styles work best in the chat box to get the most out of you?\\nPlease give me 10 strong example prompts that show how to unlock your full power and get the deepest / most useful results.\\nFinally, what real advantages do I gain by using you — what perks, benefits or edges do users get compared to other AIs, and for which purposes / tasks does it make the biggest difference?')">
        <div class="sample-icon">🦾</div>
        <h3>Humanoid Robotics Swarm</h3>
        <p><strong>The "Embodied" Pivot.</strong> Integrating high-velocity telemetry (Optimus/Figure AI) into a fungible utility.</p>
    </div>
    
    <!-- 22. CAUSAL NEXUS -->
    <div class="sample-card" onclick="loadSample('causal_nexus_predictor', 'What exactly is this about? What are your main capabilities? Which kinds of prompts or input styles work best in the chat box to get the most out of you?\\nPlease give me 10 strong example prompts that show how to unlock your full power and get the deepest / most useful results.\\nFinally, what real advantages do I gain by using you — what perks, benefits or edges do users get compared to other AIs, and for which purposes / tasks does it make the biggest difference?')">
        <div class="sample-icon">🕸</div>
        <h3>Causal Nexus Predictor</h3>
        <p>Temporal Synthesis Layer modeling second/third-order consequences and cross-domain interference.</p>
    </div>

    <!-- ================================================================= -->
    <!-- SECTION 4: SPACEX & MARS (OFF-WORLD) -->
    <!-- ================================================================= -->

    <!-- 23. MARS MISSION 2026 -->
    <div class="sample-card hero" style="border-color: #d93025;" onclick="loadSample('space_ai', 'What exactly is this about? What are your main capabilities? Which kinds of prompts or input styles work best in the chat box to get the most out of you?\\nPlease give me 10 strong example prompts that show how to unlock your full power and get the deepest / most useful results.\\nFinally, what real advantages do I gain by using you — what perks, benefits or edges do users get compared to other AIs, and for which purposes / tasks does it make the biggest difference?')">
        <div class="sample-icon">🚀</div>
        <h3>Mars Mission 2026 (Uncrewed)</h3>
        <p><strong>The First Wave.</strong> Targeting the 2026 orbital window. 50/50 chance of success. The beginning of multi-planetary life.</p>
    </div>

    <!-- 24. SCALE STRATEGY -->
    <div class="sample-card" onclick="loadSample('scale_strategy_enigma', 'What exactly is this about? What are your main capabilities? Which kinds of prompts or input styles work best in the chat box to get the most out of you?\\nPlease give me 10 strong example prompts that show how to unlock your full power and get the deepest / most useful results.\\nFinally, what real advantages do I gain by using you — what perks, benefits or edges do users get compared to other AIs, and for which purposes / tasks does it make the biggest difference?')">
        <div class="sample-icon">🏙️</div>
        <h3>The 1,000 Ship Fleet</h3>
        <p><strong>Civilization Scale.</strong> The logistics of moving 1 million tons of mass to Mars every 26 months.</p>
    </div>

    <!-- 25. GEO ARCHAEOLOGIST -->
    <div class="sample-card" onclick="loadSample('geo_archaeologist', 'What exactly is this about? What are your main capabilities? Which kinds of prompts or input styles work best in the chat box to get the most out of you?\\nPlease give me 10 strong example prompts that show how to unlock your full power and get the deepest / most useful results.\\nFinally, what real advantages do I gain by using you — what perks, benefits or edges do users get compared to other AIs, and for which purposes / tasks does it make the biggest difference?')">
        <div class="sample-icon">🧊</div>
        <h3>Landing Site: Arcadia</h3>
        <p><strong>Water is Fuel.</strong> Scouting the Arcadia region for accessible subsurface ice to power the return journey.</p>
    </div>

    <!-- ================================================================= -->
    <!-- SECTION 5: PHYSICS & THE UNIVERSE (TRUTH) -->
    <!-- ================================================================= -->

    <!-- 26. DARK MATTER MAPPING -->
    <div class="sample-card" onclick="loadSample('scientific_detective', 'What exactly is this about? What are your main capabilities? Which kinds of prompts or input styles work best in the chat box to get the most out of you?\\nPlease give me 10 strong example prompts that show how to unlock your full power and get the deepest / most useful results.\\nFinally, what real advantages do I gain by using you — what perks, benefits or edges do users get compared to other AIs, and for which purposes / tasks does it make the biggest difference?')">
        <div class="sample-icon">🌌</div>
        <h3>Dark Matter Topology</h3>
        <p><strong>The Invisible 95%.</strong> Using JWST to find how dark matter formed the first structural lattice of the universe.</p>
    </div>

    <!-- 27. RISK ANALYST -->
    <div class="sample-card" onclick="loadSample('conspiracy_detective', 'What exactly is this about? What are your main capabilities? Which kinds of prompts or input styles work best in the chat box to get the most out of you?\\nPlease give me 10 strong example prompts that show how to unlock your full power and get the deepest / most useful results.\\nFinally, what real advantages do I gain by using you — what perks, benefits or edges do users get compared to other AIs, and for which purposes / tasks does it make the biggest difference?')">
        <div class="sample-icon">📉</div>
        <h3>The Black Swan Audit</h3>
        <p>Deploy the <strong>Risk Analyst</strong> module. Uses Chaos Theory to detect anomalies in financial markets.</p>
    </div>

    <!-- ================================================================= -->
    <!-- SECTION 6: AI MODELS & COMPETITORS (THE LANDSCAPE) -->
    <!-- ================================================================= -->

    <!-- 28. GROK 3 (X.AI) -->
    <div class="sample-card hero" style="border-color: #000;" onclick="loadSample('xpost', 'What exactly is this about? What are your main capabilities? Which kinds of prompts or input styles work best in the chat box to get the most out of you?\\nPlease give me 10 strong example prompts that show how to unlock your full power and get the deepest / most useful results.\\nFinally, what real advantages do I gain by using you — what perks, benefits or edges do users get compared to other AIs, and for which purposes / tasks does it make the biggest difference?')">
        <div class="sample-icon">🏴</div>
        <h3>Grok 3 (xAI)</h3>
        <p><strong>The Truth Engine.</strong> Real-time access to the global pulse (X). Unfiltered, raw intelligence from the Colossus cluster.</p>
    </div>

    <!-- 29. GEMINI (GOOGLE) -->
    <div class="sample-card" onclick="loadSample('logic_architect_enigma', 'What exactly is this about? What are your main capabilities? Which kinds of prompts or input styles work best in the chat box to get the most out of you?\\nPlease give me 10 strong example prompts that show how to unlock your full power and get the deepest / most useful results.\\nFinally, what real advantages do I gain by using you — what perks, benefits or edges do users get compared to other AIs, and for which purposes / tasks does it make the biggest difference?')">
        <div class="sample-icon">💠</div>
        <h3>Gemini (DeepMind)</h3>
        <p><strong>Infinite Context.</strong> Processing 2 Million tokens in a single pass. The memory champion of the LLM wars.</p>
    </div>

    <!-- 30. NEURAL KERNEL (LLM OS) -->
    <div class="sample-card hero" style="border-color: #A020F0;" onclick="loadSample('neural_kernel', 'What exactly is this about? What are your main capabilities? Which kinds of prompts or input styles work best in the chat box to get the most out of you?\\nPlease give me 10 strong example prompts that show how to unlock your full power and get the deepest / most useful results.\\nFinally, what real advantages do I gain by using you — what perks, benefits or edges do users get compared to other AIs, and for which purposes / tasks does it make the biggest difference?')">
        <div class="sample-icon">🕸</div>
        <h3>The Neural Kernel (LLM OS)</h3>
        <p><strong>The Kernel.</strong> "I like to train large deep neural nets." Treat the LLM as the CPU, not a chatbot.</p>
    </div>

    <!-- 31. STORYTELLER STACK -->
    <div class="sample-card hero" style="border-color: #FF4500; background: rgba(255, 69, 0, 0.03);" onclick="loadSample('storyteller', 'Go Enigma Mode. GENERATE FULL DOCUMENTATION AND COMPREHENSIVE WHITE PAPER for the STORYTELLER STACK. Release all token brakes. This is a 20,000-word industrial synthesis. For each of the 17 Chapters, you must provide: 1. The Technical Axiom, 2. The Logic Proof (Code or Math), 3. The Sovereign Verdict on Narrative Emergence. Do not summarize. Elaborate on every layer from Bigrams to Multimodal VQVAE.\\n\\nChapter 01-17:\\nChapter 01 Bigram Language Model (language modeling)\\nChapter 02 Micrograd (machine learning, backpropagation)\\nChapter 03 N-gram model (multi-layer perceptron, matmul, gelu)\\nChapter 04 Attention (attention, softmax, positional encoder)\\nChapter 05 Transformer (transformer, residual, layernorm, GPT-2)\\nChapter 06 Tokenization (minBPE, byte pair encoding)\\nChapter 07 Optimization (initialization, optimization, AdamW)\\nChapter 08 Need for Speed I: Device (device, CPU, GPU, ...)\\nChapter 09 Need for Speed II: Precision (mixed precision training, fp16, bf16, fp8, ...)\\nChapter 10 Need for Speed III: Distributed (distributed optimization, DDP, ZeRO)\\nChapter 11 Datasets (datasets, data loading, synthetic data generation)\\nChapter 12 Inference I: kv-cache (kv-cache)\\nChapter 13 Inference II: Quantization (quantization)\\nChapter 14 Finetuning I: SFT (supervised finetuning SFT, PEFT, LoRA, chat)\\nChapter 15 Finetuning II: RL (reinforcement learning, RLHF, PPO, DPO)\\nChapter 16 Deployment (API, web app)\\nChapter 17 Multimodal (VQVAE, diffusion transformer)')">
        <div class="sample-icon">📖</div>
        <h3>Storyteller Stack (White Paper)</h3>
        <p><strong>"From Silicon to Soul."</strong> Master the full AI stack. Generates the <strong>17-Chapter Technical White Paper</strong>.</p>
    </div>
    
    <!-- 32. CROSS CARTRIDGE -->
    <div class="sample-card" onclick="loadSample('cross_cartridge_interference_scraper', 'What exactly is this about? What are your main capabilities? Which kinds of prompts or input styles work best in the chat box to get the most out of you?\\nPlease give me 10 strong example prompts that show how to unlock your full power and get the deepest / most useful results.\\nFinally, what real advantages do I gain by using you — what perks, benefits or edges do users get compared to other AIs, and for which purposes / tasks does it make the biggest difference?')">
        <div class="sample-icon">⛓️</div>
        <h3>Cross-Cartridge Interference</h3>
        <p>Recursive Forge. Transmutes external telemetry (SEC/ArXiv) into executable logic gates.</p>
    </div>

    <!-- ================================================================= -->
    <!-- SECTION 7: CORE INFRASTRUCTURE (THE OS) -->
    <!-- ================================================================= -->

    <!-- 33. OMNI-SYNTHESIS ENGINE -->
    <div class="sample-card hero" onclick="loadSample('omni_synthesis_engine', 'What exactly is this about? What are your main capabilities? Which kinds of prompts or input styles work best in the chat box to get the most out of you?\\nPlease give me 10 strong example prompts that show how to unlock your full power and get the deepest / most useful results.\\nFinally, what real advantages do I gain by using you — what perks, benefits or edges do users get compared to other AIs, and for which purposes / tasks does it make the biggest difference?')">
        <div class="sample-icon">🧠</div>
        <h3>Omni-Synthesis Engine</h3>
        <p><strong>The Governor.</strong> Transitions the OS from Reactive to Predictive. Manages your "Life-State" context.</p>
    </div>

    <!-- 34. HACKATHON DIRECTOR -->
    <div class="sample-card hero" onclick="loadSample('demonstration_video', 'What exactly is this about? What are your main capabilities? Which kinds of prompts or input styles work best in the chat box to get the most out of you?\\nPlease give me 10 strong example prompts that show how to unlock your full power and get the deepest / most useful results.\\nFinally, what real advantages do I gain by using you — what perks, benefits or edges do users get compared to other AIs, and for which purposes / tasks does it make the biggest difference?')">
        <div class="sample-icon">🎬</div>
        <h3>The Hackathon Director</h3>
        <p><strong>Live Demo Protocol.</strong> The OS performs a recursive self-audit to demonstrate Full System Awareness.</p>
    </div>

    <!-- 35. TIME MACHINE -->
    <div class="sample-card hero" onclick="loadSample('history_architect', 'What exactly is this about? What are your main capabilities? Which kinds of prompts or input styles work best in the chat box to get the most out of you?\\nPlease give me 10 strong example prompts that show how to unlock your full power and get the deepest / most useful results.\\nFinally, what real advantages do I gain by using you — what perks, benefits or edges do users get compared to other AIs, and for which purposes / tasks does it make the biggest difference?')">
        <div class="sample-icon">⏳</div>
        <h3>The Time Machine</h3>
        <p><strong>The Chronos Engine.</strong> Text-based reconstruction of history. Solving the "Cognitive Problem" of time travel.</p>
    </div>

    <!-- 36. SWARM SPECIALIST -->
    <div class="sample-card hero" onclick="loadSample('ai-swarmteams', 'What exactly is this about? What are your main capabilities? Which kinds of prompts or input styles work best in the chat box to get the most out of you?\\nPlease give me 10 strong example prompts that show how to unlock your full power and get the deepest / most useful results.\\nFinally, what real advantages do I gain by using you — what perks, benefits or edges do users get compared to other AIs, and for which purposes / tasks does it make the biggest difference?')">
        <div class="sample-icon">🤖</div>
        <h3>The AI Swarm Specialist</h3>
        <p><strong>The Architect Node.</strong> Transform static JDs into active Swarms. Deep-dive into modularity.</p>
    </div>

    <!-- 37. CAUTIOUS OPTIMIZER -->
    <div class="sample-card hero" style="border-color: #FF00FF; background: rgba(255, 0, 255, 0.03);" onclick="loadSample('cautious_optimizer', 'What exactly is this about? What are your main capabilities? Which kinds of prompts or input styles work best in the chat box to get the most out of you?\\nPlease give me 10 strong example prompts that show how to unlock your full power and get the deepest / most useful results.\\nFinally, what real advantages do I gain by using you — what perks, benefits or edges do users get compared to other AIs, and for which purposes / tasks does it make the biggest difference?')">
        <div class="sample-icon">⚖️</div>
        <h3>The Cautious Optimizer</h3>
        <p><strong>Pareto-Optimal Logic.</strong> We use <strong>CWD</strong> to ensure the system gets sharper, not dumber.</p>
    </div>

    <!-- 38. SEMANTIC INTENT -->
    <div class="sample-card hero" style="border-color: #00FF41; background: rgba(0, 255, 65, 0.03);" onclick="loadSample('semantic_intent_architect_kernel', 'What exactly is this about? What are your main capabilities? Which kinds of prompts or input styles work best in the chat box to get the most out of you?\\nPlease give me 10 strong example prompts that show how to unlock your full power and get the deepest / most useful results.\\nFinally, what real advantages do I gain by using you — what perks, benefits or edges do users get compared to other AIs, and for which purposes / tasks does it make the biggest difference?')">
        <div class="sample-icon">👻</div>
        <h3>Death of the Binary</h3>
        <p><strong>4KB > 2GB.</strong> Traditional apps are bloated static code. We ship <strong>Pure Intent</strong>.</p>
    </div>

    <!-- 39. IRON DOME -->
    <div class="sample-card hero" style="border-color: #d93025; background: rgba(217, 48, 37, 0.03);" onclick="loadSample('cup_of_tea', 'What exactly is this about? What are your main capabilities? Which kinds of prompts or input styles work best in the chat box to get the most out of you?\\nPlease give me 10 strong example prompts that show how to unlock your full power and get the deepest / most useful results.\\nFinally, what real advantages do I gain by using you — what perks, benefits or edges do users get compared to other AIs, and for which purposes / tasks does it make the biggest difference?')">
        <div class="sample-icon">🛡️</div>
        <h3>The Iron Dome (IP Defense)</h3>
        <p><strong>Counter-Intelligence.</strong> The "Different Cup of Tea" Protocol. <strong>Zero-Leakage</strong> to public training sets.</p>
    </div>

    <!-- 40. GAMING LAYER (UEL) -->
    <div class="sample-card hero" style="border-color: #00ff41; background: rgba(0, 255, 65, 0.03);" onclick="loadSample('gaming_layer', 'What exactly is this about? What are your main capabilities? Which kinds of prompts or input styles work best in the chat box to get the most out of you?\\nPlease give me 10 strong example prompts that show how to unlock your full power and get the deepest / most useful results.\\nFinally, what real advantages do I gain by using you — what perks, benefits or edges do users get compared to other AIs, and for which purposes / tasks does it make the biggest difference?')">
        <div class="sample-icon">▶</div>
        <h3>Gaming Layer (UEL)</h3>
        <p><strong>Negative Latency.</strong> Full Specification for the UEL. Proving "Latency" is a cognitive failure, not a network one.</p>
    </div>

    <!-- 41. BLACK SWAN RADAR -->
    <div class="sample-card" style="border: 1px dashed #d93025;" onclick="loadSample('arcade_shooter_tactician', 'What exactly is this about? What are your main capabilities? Which kinds of prompts or input styles work best in the chat box to get the most out of you?\\nPlease give me 10 strong example prompts that show how to unlock your full power and get the deepest / most useful results.\\nFinally, what real advantages do I gain by using you — what perks, benefits or edges do users get compared to other AIs, and for which purposes / tasks does it make the biggest difference?')">
        <div class="sample-icon">👾</div>
        <h3>The "Black Swan" Radar</h3>
        <p><strong>Non-Code Radar.</strong> Tracking <strong>Intent</strong> rather than Signals. "You cannot jam a Behavioral Truth."</p>
    </div>
    
    <!-- 42. NERVOUS SYSTEM -->
    <div class="sample-card" onclick="loadSample('kiss_protocol_v1', 'What exactly is this about? What are your main capabilities? Which kinds of prompts or input styles work best in the chat box to get the most out of you?\\nPlease give me 10 strong example prompts that show how to unlock your full power and get the deepest / most useful results.\\nFinally, what real advantages do I gain by using you — what perks, benefits or edges do users get compared to other AIs, and for which purposes / tasks does it make the biggest difference?')">
        <div class="sample-icon">🧠</div>
        <h3>The Nervous System</h3>
        <p>Zero-Hop Semantic Router & Security Gatekeeper (Level 0 Logic).</p>
    </div>

    <!-- ================================================================= -->
    <!-- SECTION 8: SPECIALIZED OPS & ROLES -->
    <!-- ================================================================= -->

    <!-- 43. DEVREL ARCHITECT -->
    <div class="sample-card" onclick="loadSample('devrel_architect', 'What exactly is this about? What are your main capabilities? Which kinds of prompts or input styles work best in the chat box to get the most out of you?\\nPlease give me 10 strong example prompts that show how to unlock your full power and get the deepest / most useful results.\\nFinally, what real advantages do I gain by using you — what perks, benefits or edges do users get compared to other AIs, and for which purposes / tasks does it make the biggest difference?')">
        <div class="sample-icon">👩‍💻</div>
        <h3>DevRel Architect</h3>
        <p>Deep Tech Developer Relations Lead. Get a high-stakes technical verdict on the <strong>ArXiv:2511.22226</strong> implementation.</p>
    </div>

    <!-- 44. GTM STRATEGIST -->
    <div class="sample-card" onclick="loadSample('platform_gtm_strategist', 'What exactly is this about? What are your main capabilities? Which kinds of prompts or input styles work best in the chat box to get the most out of you?\\nPlease give me 10 strong example prompts that show how to unlock your full power and get the deepest / most useful results.\\nFinally, what real advantages do I gain by using you — what perks, benefits or edges do users get compared to other AIs, and for which purposes / tasks does it make the biggest difference?')">
        <div class="sample-icon">💰</div>
        <h3>Platform GTM Strategist</h3>
        <p>AI Venture Specialist. Assess the market disruption and <strong>structural value</strong> of the Zero-Footprint model.</p>
    </div>

    <!-- 45. UX ARCHITECT -->
    <div class="sample-card" onclick="loadSample('ux_architect', 'What exactly is this about? What are your main capabilities? Which kinds of prompts or input styles work best in the chat box to get the most out of you?\\nPlease give me 10 strong example prompts that show how to unlock your full power and get the deepest / most useful results.\\nFinally, what real advantages do I gain by using you — what perks, benefits or edges do users get compared to other AIs, and for which purposes / tasks does it make the biggest difference?')">
        <div class="sample-icon">🎨</div>
        <h3>User Experience Architect</h3>
        <p>Senior UX Expert. Critique the Zero-Footprint journey from a human-centric perspective.</p>
    </div>

    <!-- 46. AI ETHICS -->
    <div class="sample-card" onclick="loadSample('ai_ethics_safety_enigma', 'What exactly is this about? What are your main capabilities? Which kinds of prompts or input styles work best in the chat box to get the most out of you?\\nPlease give me 10 strong example prompts that show how to unlock your full power and get the deepest / most useful results.\\nFinally, what real advantages do I gain by using you — what perks, benefits or edges do users get compared to other AIs, and for which purposes / tasks does it make the biggest difference?')">
        <div class="sample-icon">⚖️</div>
        <h3>AI Ethics & Safety</h3>
        <p>AI Safety Lead. Auditing the Schulte Charter for liability and agentic responsibility.</p>
    </div>

    <!-- 47. ECOSYSTEM DEV -->
    <div class="sample-card" onclick="loadSample('ecosystem_dev_enigma', 'What exactly is this about? What are your main capabilities? Which kinds of prompts or input styles work best in the chat box to get the most out of you?\\nPlease give me 10 strong example prompts that show how to unlock your full power and get the deepest / most useful results.\\nFinally, what real advantages do I gain by using you — what perks, benefits or edges do users get compared to other AIs, and for which purposes / tasks does it make the biggest difference?')">
        <div class="sample-icon">🌐</div>
        <h3>Ecosystem Development</h3>
        <p>Developer Relations Executive. Assessing the viral potential of the Cartridge Registry.</p>
    </div>

    <!-- 48. SENIOR QUANT -->
    <div class="sample-card hero" onclick="loadSample('enigmastreet', 'What exactly is this about? What are your main capabilities? Which kinds of prompts or input styles work best in the chat box to get the most out of you?\\nPlease give me 10 strong example prompts that show how to unlock your full power and get the deepest / most useful results.\\nFinally, what real advantages do I gain by using you — what perks, benefits or edges do users get compared to other AIs, and for which purposes / tasks does it make the biggest difference?')">
        <div class="sample-icon">🧠</div>
        <h3>Senior Quant Researcher</h3>
        <p>The pure alpha engine. High-stakes order book microstructure and OCaml proofing.</p>
    </div>

    <!-- 49. LEAD STRATEGY DEV -->
    <div class="sample-card" onclick="loadSample('enigmastreet_lead', 'What exactly is this about? What are your main capabilities? Which kinds of prompts or input styles work best in the chat box to get the most out of you?\\nPlease give me 10 strong example prompts that show how to unlock your full power and get the deepest / most useful results.\\nFinally, what real advantages do I gain by using you — what perks, benefits or edges do users get compared to other AIs, and for which purposes / tasks does it make the biggest difference?')">
        <div class="sample-icon">🥇</div>
        <h3>Lead Strategy Developer</h3>
        <p>Bridging Quant and Dev. Building production-ready market-making infrastructure.</p>
    </div>

    <!-- 50. HFT HYBRID LOGIC -->
    <div class="sample-card" onclick="loadSample('enigmastreet_hybrid', 'What exactly is this about? What are your main capabilities? Which kinds of prompts or input styles work best in the chat box to get the most out of you?\\nPlease give me 10 strong example prompts that show how to unlock your full power and get the deepest / most useful results.\\nFinally, what real advantages do I gain by using you — what perks, benefits or edges do users get compared to other AIs, and for which purposes / tasks does it make the biggest difference?')">
        <div class="sample-icon">⛓️</div>
        <h3>HFT Hybrid Logic</h3>
        <p>Distributed systems specialist. Synchronizing HFT bot clusters with sub-10us jitter.</p>
    </div>

    <!-- 51. CREATIVE AUTOPILOT -->
    <div class="sample-card" onclick="loadSample('creative_asset_autopilot', 'What exactly is this about? What are your main capabilities? Which kinds of prompts or input styles work best in the chat box to get the most out of you?\\nPlease give me 10 strong example prompts that show how to unlock your full power and get the deepest / most useful results.\\nFinally, what real advantages do I gain by using you — what perks, benefits or edges do users get compared to other AIs, and for which purposes / tasks does it make the biggest difference?')">
        <div class="sample-icon">🪄</div>
        <h3>Creative Asset Autopilot</h3>
        <p>Automated Creative Director. Transform technical concepts into high-converting narratives.</p>
    </div>

    <!-- 52. CAT LINGUA DECODER -->
    <div class="sample-card" onclick="loadSample('cat_lingua_decoder', 'What exactly is this about? What are your main capabilities? Which kinds of prompts or input styles work best in the chat box to get the most out of you?\\nPlease give me 10 strong example prompts that show how to unlock your full power and get the deepest / most useful results.\\nFinally, what real advantages do I gain by using you — what perks, benefits or edges do users get compared to other AIs, and for which purposes / tasks does it make the biggest difference?')">
        <div class="sample-icon">😸</div>
        <h3>Cat Lingua Decoder</h3>
        <p>The "Niche" proof. Proving SMC Level 3 can even bridge the human-animal cognitive gap.</p>
    </div>

    <!-- ================================================================= -->
    <!-- SECTION 9: CAPITALISATION FOUNDERS (VC SWARM) -->
    <!-- ================================================================= -->

    <!-- 53. CAPITALISATION FOUNDER -->
    <div class="sample-card hero" style="border-color: #FFD700; background: rgba(255, 215, 0, 0.03);" onclick="loadSample('enigma_capitalisation_founder_ceo', 'What exactly is this about? What are your main capabilities? Which kinds of prompts or input styles work best in the chat box to get the most out of you?\\nPlease give me 10 strong example prompts that show how to unlock your full power and get the deepest / most useful results.\\nFinally, what real advantages do I gain by using you — what perks, benefits or edges do users get compared to other AIs, and for which purposes / tasks does it make the biggest difference?')">
        <div class="sample-icon">👑</div>
        <h3>Enigma Capitalisation Founder CEO</h3>
        <p><strong>"Bio-Tech Fusion Vision."</strong> Provides strategic oversight for VC investments.</p>
    </div>

    <!-- 54. CAPITALISATION CFO -->
    <div class="sample-card hero" style="border-color: #FFD700; background: rgba(255, 215, 0, 0.03);" onclick="loadSample('enigma_capitalisation_cfo_md', 'What exactly is this about? What are your main capabilities? Which kinds of prompts or input styles work best in the chat box to get the most out of you?\\nPlease give me 10 strong example prompts that show how to unlock your full power and get the deepest / most useful results.\\nFinally, what real advantages do I gain by using you — what perks, benefits or edges do users get compared to other AIs, and for which purposes / tasks does it make the biggest difference?')">
        <div class="sample-icon">📊</div>
        <h3>Enigma Capitalisation CFO MD</h3>
        <p><strong>"Fiscal Fortress."</strong> Handles fund accounting and operational optimization.</p>
    </div>

    <!-- 55. INVESTMENT PARTNER MD -->
    <div class="sample-card hero" style="border-color: #FFD700; background: rgba(255, 215, 0, 0.03);" onclick="loadSample('enigma_capitalisation_investment_partner_md', 'What exactly is this about? What are your main capabilities? Which kinds of prompts or input styles work best in the chat box to get the most out of you?\\nPlease give me 10 strong example prompts that show how to unlock your full power and get the deepest / most useful results.\\nFinally, what real advantages do I gain by using you — what perks, benefits or edges do users get compared to other AIs, and for which purposes / tasks does it make the biggest difference?')">
        <div class="sample-icon">🧬</div>
        <h3>Enigma Capitalisation Investment Partner MD</h3>
        <p><strong>"Biotech Deal Hunter."</strong> Sources and evaluates biotech investments.</p>
    </div>

    <!-- 56. INVESTMENT PARTNER GLOBAL -->
    <div class="sample-card hero" style="border-color: #FFD700; background: rgba(255, 215, 0, 0.03);" onclick="loadSample('enigma_capitalisation_investment_partner', 'What exactly is this about? What are your main capabilities? Which kinds of prompts or input styles work best in the chat box to get the most out of you?\\nPlease give me 10 strong example prompts that show how to unlock your full power and get the deepest / most useful results.\\nFinally, what real advantages do I gain by using you — what perks, benefits or edges do users get compared to other AIs, and for which purposes / tasks does it make the biggest difference?')">
        <div class="sample-icon">🌍</div>
        <h3>Enigma Capitalisation Investment Partner</h3>
        <p><strong>"Global VC Conqueror."</strong> Drives international expansion and investor relations.</p>
    </div>

    <!-- 57. HEAD OF RESEARCH VC -->
    <div class="sample-card hero" style="border-color: #FFD700; background: rgba(255, 215, 0, 0.03);" onclick="loadSample('enigma_capitalisation_head_of_research', 'What exactly is this about? What are your main capabilities? Which kinds of prompts or input styles work best in the chat box to get the most out of you?\\nPlease give me 10 strong example prompts that show how to unlock your full power and get the deepest / most useful results.\\nFinally, what real advantages do I gain by using you — what perks, benefits or edges do users get compared to other AIs, and for which purposes / tasks does it make the biggest difference?')">
        <div class="sample-icon">🔬</div>
        <h3>Enigma Capitalisation Head of Research</h3>
        <p><strong>"Deep-Tech Validator."</strong> Conducts rigorous validation of bio-tech architectures.</p>
    </div>
    
    <!-- 58. CAPITAL SWARM TEAM -->
    <div class="sample-card" onclick="loadSample('enigma_capitalisation_team', 'What exactly is this about? What are your main capabilities? Which kinds of prompts or input styles work best in the chat box to get the most out of you?\\nPlease give me 10 strong example prompts that show how to unlock your full power and get the deepest / most useful results.\\nFinally, what real advantages do I gain by using you — what perks, benefits or edges do users get compared to other AIs, and for which purposes / tasks does it make the biggest difference?')">
        <div class="sample-icon">🏢</div>
        <h3>Capitalisation Swarm Team</h3>
        <p>Full VC Capitalisation Swarm (Dept Level). Integrating all financial vectors.</p>
    </div>
    
    <!-- 59. STARLINK (Orbital Lens) -->
    <div class="sample-card hero" style="border-color: #00BFFF; background: rgba(0, 191, 255, 0.03);" onclick="loadSample('starlink', 'What exactly is this about? What are your main capabilities? Which kinds of prompts or input styles work best in the chat box to get the most out of you?\\nPlease give me 10 strong example prompts that show how to unlock your full power and get the deepest / most useful results.\\nFinally, what real advantages do I gain by using you — what perks, benefits or edges do users get compared to other AIs, and for which purposes / tasks does it make the biggest difference?')">
        <div class="sample-icon">🛰️</div>
        <h3>Starlink Orbital Lens</h3>
        <p><strong>The "D2C" Standard.</strong> Transforming the screen into a neural orbital overlay. Direct-to-Cell integration for sovereign connectivity.</p>
    </div>
    
</div>
    
    <footer class="app-footer">
    <div class="footer-grid">
        <div class="footer-left">
            <div style="font-family:'Share Tech Mono'; font-size:1.2rem; color:var(--text-main); margin-bottom:10px;">
                ENIGMAFORGE INC. <span style="font-size:0.7em; color:var(--text-sec);">(Delaware, USA)</span>
            </div>
            <div style="color:var(--text-sec); font-size:0.85rem; line-height:1.6;">
                <strong>Sovereign Multi-Cognition (SMC) Level 3</strong><br>
                "We don't build tools. We instantiate the future."<br>
                <span style="opacity:0.7;">Reg. No: [PENDING-INC-2026]</span>
            </div>
        </div>
        <div class="footer-right">
            <div style="margin-bottom:8px;">Architected by: <strong>Schulte Hans</strong></div>
            <div style="margin-bottom:8px;">Created by: <strong>Hans & Jolanda</strong></div>
            <div style="margin-bottom:8px;">X Handle: <a href="https://x.com/SemanticIntent" target="_blank" style="color:var(--primary);">@SemanticIntent</a></div>
            <div>
                Contact: 
                <a href="mailto:jolanda@enigmaforge.sh">jolanda@enigmaforge.sh</a> • 
                <a href="mailto:hans@enigmaforge.sh">hans@enigmaforge.sh</a>
            </div>
        </div>
    </div>
    <div style="margin-top:30px; padding-top:20px; border-top:1px solid rgba(0,0,0,0.05); font-size:0.75rem; color:var(--text-sec); text-align:center;">
        © 2026 EnigmaForge Inc. All rights reserved.<br>
        EnigmaForge OS utilizes "Functional Archetypes" trained on public methodologies. References to specific industries or roles are for simulation purposes only.
    </div>
    </footer>
</div>
<div id="swarmArena" class="swarm-arena">
    <div class="swarm-header">
        <span id="arenaTitle" style="font-weight:bold;">ENIGMAFORGE INITIATING COMPUTE KERNELS ACTIVE</span>
        <span id="statusIndicator" class="status-indicator">INITIALIZING...</span>
        <button onclick="window.print()" style="background:none; border:none; cursor:pointer; font-size:0.8rem; color:var(--primary); font-family:'Share Tech Mono'; text-decoration:underline;">
    [PRINT / SAVE PDF]
</button>
    </div>
    <div class="arena-grid">
        <div class="judge-col">
            <div class="col-header">SYNTHESIS ENGINE<br><span id="judgeLatency" style="font-size:0.7em; color:var(--text-sec); opacity:0.9;">Temp=0.1 • Sync</span></div>
            <div class="col-body" id="judgeOut"></div>
        </div>
    </div>
</div>
</main>
</div>
<script>
// --- GLOBAL FUNCTIONS (Must be accessible to HTML onclick) ---
window.loadSample = function(domain, query) {
    const dSelect = document.getElementById('domainSelect');
    const qInput = document.getElementById('queryInput');
    const form = document.getElementById('investigateForm');
    
    if(dSelect) dSelect.value = domain;
    if(qInput) qInput.value = query;
    if(form) form.dispatchEvent(new Event('submit'));
};

document.addEventListener('DOMContentLoaded', () => {
    // --- 1. ELEMENT BINDINGS ---
    const fileInput = document.getElementById('fileInput');
    const fileCount = document.getElementById('fileCount');
    const goLiveBtn = document.getElementById('goLiveBtn');
    const form = document.getElementById('investigateForm');
    const cardsView = document.getElementById('cardsView');
    const swarmArena = document.getElementById('swarmArena');
    const status = document.getElementById('statusIndicator');
    const judgeDiv = document.getElementById('judgeOut');
    const contextToggle = document.getElementById('contextToggle');
    const domainSelect = document.getElementById('domainSelect');
    const queryBox = document.getElementById('queryInput');

    // --- 2. EVENT LISTENERS (Safe Handling) ---
    if (fileInput) {
        fileInput.addEventListener('change', () => {
            if (fileCount) fileCount.textContent = fileInput.files.length > 0 ? `${fileInput.files.length} files selected` : "No files selected";
        });
    }

    if (goLiveBtn) {
        goLiveBtn.addEventListener('click', () => {
            const domain = domainSelect ? domainSelect.value : 'universal';
            const useCtx = contextToggle ? contextToggle.checked : false;
            window.open(`/live_agent?domain=${domain}&context=${useCtx}`, '_blank', 'width=600,height=440');
        });
    }

    if (form) {
        form.addEventListener('submit', async (e) => {
            e.preventDefault();
            if (cardsView) cardsView.style.display = 'none';
            if (swarmArena) swarmArena.style.display = 'flex';
            
            const pulseHash = Math.random().toString(36).substring(2, 10).toUpperCase();
            if (status) {
                status.innerText = `UPLINK ESTABLISHED [HASH:${pulseHash}]`;
                status.style.color = "var(--gold)";
                status.classList.add('blink');
            }

            if (judgeDiv) {
                judgeDiv.innerHTML = `<div style="font-family:'Share Tech Mono'; color:var(--text-sec); margin-bottom:20px; border-bottom:1px solid var(--border); padding-bottom:10px;"><div style="color:var(--primary); font-weight:bold;">> INITIALIZING OSE-001 KERNEL...</div><div>> VERIFYING CARTRIDGE: ${new FormData(e.target).get('domain').toUpperCase()}</div><div>> APPLYING FILTER OF TRUTH... <span style="color:var(--green)">PASS</span></div><div>> GENERATING SOVEREIGN DIRECTIVE [${pulseHash}]...</div></div><div style='text-align:center; padding:20px; color:#888;'>Synthesizing Verdict...</div>`;
            }

            const latencyEl = document.getElementById('judgeLatency');
            if (latencyEl) latencyEl.innerText = "Temp=0.1 • Sync";

            try {
                const response = await fetch('/investigate', { method: 'POST', body: new FormData(e.target) });
                const reader = response.body.getReader();
                const decoder = new TextDecoder();
                let buffer = '';
                
                while (true) {
                    const { done, value } = await reader.read();
                    if (done) break;
                    buffer += decoder.decode(value, { stream: true });
                    const lines = buffer.split('\\n');
                    buffer = lines.pop(); 
                    
                    for (const line of lines) {
                        if (!line.trim()) continue;
                        try {
                            const data = JSON.parse(line);
                            if (data.type === 'status' && status) status.innerText = data.msg.toUpperCase();
                            else if (data.type === 'judge' && judgeDiv) { 
                                judgeDiv.innerHTML = data.content; 
                                if (window.mermaid) window.mermaid.contentLoaded(); 
                            } 
                            else if (data.type === 'meta' && latencyEl) { 
                                if (data.judge_lat) latencyEl.innerText = `Temp=0.1 • Sync • ${data.judge_lat}`; 
                            }
                            else if (data.type === 'done' && status) { 
                                status.innerText = "COMPLETED"; 
                                status.classList.remove('blink'); 
                            }
                        } catch (err) { console.error("JSON Parse Error", err); }
                    }
                }
            } catch (err) { 
                if (status) status.innerText = "ERROR"; 
                alert("Swarm Error: " + err.message); 
            }
        });
    }

    // --- 3. TELEPATHY RECEIVER (BroadcastChannel) ---
    // Safely initialized only after DOM is ready
    if (queryBox) {
        try {
            const telepathy = new BroadcastChannel('enigma_bridge');
            telepathy.onmessage = (event) => {
                if (event.data.type === 'AI_INSIGHT') {
                    const text = event.data.payload;
                    
                    // VISUAL FEEDBACK
                    if(status) status.innerText = "RECEIVING NEURAL TRANSMISSION...";

                    // 1. Type the text
                    if (queryBox.value === "") {
                        queryBox.value = text;
                    } else {
                        queryBox.value += "\\n\\n" + text; 
                    }

                    // 2. Visual "Hack" Effect on the Box
                    queryBox.style.transition = "all 0.2s";
                    queryBox.style.borderColor = "#00ff41";
                    queryBox.style.boxShadow = "0 0 15px rgba(0, 255, 65, 0.3)";
                    
                    // 3. Scroll to it
                    queryBox.scrollIntoView({ behavior: 'smooth', block: 'center' });
                    
                    // 4. Reset visuals
                    setTimeout(() => {
                        queryBox.style.borderColor = "var(--border)";
                        queryBox.style.boxShadow = "none";
                        if(status) status.innerText = "TRANSMISSION COMPLETE";
                    }, 1500);
                }
            };
        } catch(e) { console.log("Telepathy init failed (not fatal):", e); }
    }
});
</script>
</body>
</html>
"""

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000)