#!/bin/bash

# Needle in a Haystack Test Runner Script
# This script provides easy commands to run various needle in haystack tests

echo "=========================================="
echo "    LLM Needle in a Haystack Tester"
echo "=========================================="

# Default parameters
MODEL="meta-llama/Meta-Llama-3.1-8B-Instruct"
DEVICE="cuda:0"
PRESS="expected_attention"
COMPRESSION="0.1"

# Function to run quick test
run_quick_test() {
    echo "Running quick needle test..."
    python LLMTest_NeedleInAHaystack.py \
        --model="$MODEL" \
        --device="$DEVICE" \
        --press_name="$PRESS" \
        --compression_ratio=$COMPRESSION \
        --context_lengths="[1000, 2000, 4000]" \
        --needle_positions="[0.0, 0.5, 1.0]" \
        --num_needles=2
}

# Function to run comprehensive test
run_comprehensive_test() {
    echo "Running comprehensive needle test..."
    python LLMTest_NeedleInAHaystack.py \
        --model="$MODEL" \
        --device="$DEVICE" \
        --press_name="$PRESS" \
        --compression_ratio=$COMPRESSION \
        --context_lengths="[1000, 2000, 4000, 8000, 16000, 32000]" \
        --needle_positions="[0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0]" \
        --num_needles=3
}

# Function to run custom test
run_custom_test() {
    echo "Running custom needle test..."
    echo "You can modify the parameters below:"
    echo "Current settings:"
    echo "  Model: $MODEL"
    echo "  Device: $DEVICE" 
    echo "  Press: $PRESS"
    echo "  Compression: $COMPRESSION"
    echo ""
    
    python LLMTest_NeedleInAHaystack.py \
        --model="$MODEL" \
        --device="$DEVICE" \
        --press_name="$PRESS" \
        --compression_ratio=$COMPRESSION \
        "$@"
}

# Function to test different compression methods
test_compression_methods() {
    echo "Testing different compression methods..."
    
    for press in "expected_attention" "snapkv" "streaming_llm" "full_kv"; do
        echo "Testing with press: $press"
        python LLMTest_NeedleInAHaystack.py \
            --model="$MODEL" \
            --device="$DEVICE" \
            --press_name="$press" \
            --compression_ratio=$COMPRESSION \
            --context_lengths="[2000, 8000]" \
            --needle_positions="[0.0, 0.5, 1.0]" \
            --num_needles=2
    done
}

# Function to test different compression ratios
test_compression_ratios() {
    echo "Testing different compression ratios..."
    
    for ratio in "0.05" "0.1" "0.2" "0.5"; do
        echo "Testing with compression ratio: $ratio"
        python LLMTest_NeedleInAHaystack.py \
            --model="$MODEL" \
            --device="$DEVICE" \
            --press_name="$PRESS" \
            --compression_ratio=$ratio \
            --context_lengths="[2000, 8000]" \
            --needle_positions="[0.0, 0.5, 1.0]" \
            --num_needles=2
    done
}

# Main menu
case "$1" in
    "quick")
        run_quick_test
        ;;
    "comprehensive")
        run_comprehensive_test
        ;;
    "custom")
        shift
        run_custom_test "$@"
        ;;
    "test-press")
        test_compression_methods
        ;;
    "test-ratios")
        test_compression_ratios
        ;;
    *)
        echo "Usage: $0 {quick|comprehensive|custom|test-press|test-ratios}"
        echo ""
        echo "Commands:"
        echo "  quick         - Run a quick test with basic settings"
        echo "  comprehensive - Run a comprehensive test with multiple settings"
        echo "  custom        - Run with custom parameters (pass additional args)"
        echo "  test-press    - Test different compression methods"
        echo "  test-ratios   - Test different compression ratios"
        echo ""
        echo "Examples:"
        echo "  $0 quick"
        echo "  $0 comprehensive"
        echo "  $0 custom --context_lengths='[4000, 8000]' --num_needles=5"
        echo "  $0 test-press"
        echo ""
        echo "To modify default settings, edit this script and change:"
        echo "  MODEL=\"$MODEL\""
        echo "  DEVICE=\"$DEVICE\""
        echo "  PRESS=\"$PRESS\""
        echo "  COMPRESSION=\"$COMPRESSION\""
        exit 1
        ;;
esac

echo "Test completed!"
