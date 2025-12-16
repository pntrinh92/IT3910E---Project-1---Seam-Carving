#!/bin/bash

echo "╔═══════════════════════════════════════════════════════════════╗"
echo "║                                                               ║"
echo "║              SEAM CARVING - STREAMLIT WEB DEMO                ║"
echo "║                                                               ║"
echo "╚═══════════════════════════════════════════════════════════════╝"
echo ""

if ! command -v streamlit &> /dev/null
then
    echo " Streamlit is not installed."
    echo ""
    echo "Installing required packages..."
    echo ""
    pip install -r requirements.txt
    echo ""
fi

if ! command -v streamlit &> /dev/null
then
    echo "Installation failed. Please run manually:"
    echo "   pip install -r requirements.txt"
    exit 1
fi

echo "All dependencies ready."
echo ""
echo "Starting Streamlit app..."
echo ""
echo "The app will open in your browser at: http://localhost:8501"
echo ""
echo "Tips:"
echo "   • Upload an image to get started"
echo "   • Try the 70% preset for best results"
echo "   • Compare with standard resize to see the difference"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""
echo "════════════════════════════════════════════════════════════════"
echo ""

# Launch streamlit
streamlit run streamlit_app.py
