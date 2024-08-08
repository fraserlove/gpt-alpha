# LLM Evaluation Script
# Results are stored under lm-evaluation-harness/results

# ./run_eval.sh {model1 model2 ... modelN}
# - model: HuggingFace model (i.e. fraserlove/gpt2)

# First download Eleuther LM evaluation harness
# git clone https://github.com/EleutherAI/lm-evaluation-harness/
# cd lm-evaluation-harness
# pip install -e .
# cd ..

if [ $# -eq 0 ]; then
    echo "Error: missing HuggingFace model(s)"
    echo "Usage: ./run_eval.sh hf_user/hf_model1 hf_user/hf_model2 ..."
    exit 1
fi

RESULTS_DIR="eval"
echo "Saving results to $RESULTS_DIR"

for MODEL in "$@"; do
    echo "Evaluating model $MODEL"

    # CommonSenseQA
    lm_eval --model hf --model_args pretrained=$MODEL --batch_size 16 --device cuda --tasks commonsense_qa --output_path "$RESULTS_DIR/commonsenseqa_0shot"
    # PIQA
    lm_eval --model hf --model_args pretrained=$MODEL --batch_size 16 --device cuda --tasks piqa --output_path "$RESULTS_DIR/piqa_0shot"
    # OpenBookQA
    lm_eval --model hf --model_args pretrained=$MODEL --batch_size 16 --device cuda --tasks openbookqa --output_path "$RESULTS_DIR/openbookqa_0shot"
    # TriviaQA
    lm_eval --model hf --model_args pretrained=$MODEL --batch_size 16 --device cuda --tasks triviaqa --output_path "$RESULTS_DIR/triviaqa_0shot"
    # TruthfulQA
    lm_eval --model hf --model_args pretrained=$MODEL --batch_size 16 --device cuda --tasks truthfulqa_mc1,truthfulqa_mc2 --output_path "$RESULTS_DIR/truthfulqa_0shot"
    # MMLU
    lm_eval --model hf --model_args pretrained=$MODEL --batch_size 16 --device cuda --tasks mmlu --output_path "$RESULTS_DIR/mmlu_5shot" --num_fewshot 5
    # WinoGrande
    lm_eval --model hf --model_args pretrained=$MODEL --batch_size 16 --device cuda --tasks winogrande --output_path "$RESULTS_DIR/winogrande_5shot" --num_fewshot 5
    # Arc Challange
    lm_eval --model hf --model_args pretrained=$MODEL --batch_size 16 --device cuda --tasks arc_challenge --output_path "$RESULTS_DIR/arc_challenge_25shot" --num_fewshot 25
    # HellaSwag
    lm_eval --model hf --model_args pretrained=$MODEL --batch_size 16 --device cuda --tasks hellaswag --output_path "$RESULTS_DIR/hellaswag_10shot" --num_fewshot 10
    # GSM-8K
    lm_eval --model hf --model_args pretrained=$MODEL --batch_size 16 --device cuda --tasks gsm8k --output_path "$RESULTS_DIR/gsm8k_5shot" --num_fewshot 5

    python eval_results.py $MODEL
done