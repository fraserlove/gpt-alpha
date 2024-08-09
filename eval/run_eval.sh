# LLM Evaluation Script

# ./run_eval.sh {hf_user/hf_model1} {hf_user/hf_model2} ... {hf_user/hf_modelN}

# First download Eleuther LM evaluation harness
# git clone https://github.com/EleutherAI/lm-evaluation-harness/
# cd lm-evaluation-harness
# git checkout 0571eeb14d4e48aac51956a726c62cd8b382b3d8
# pip install -e .
# cd ..

if [ $# -eq 0 ]; then
    echo "Error: missing HuggingFace model(s)"
    echo "Usage: ./run_eval.sh {hf_user/hf_model1} {hf_user/hf_model2} ... {hf_user/hf_modelN}"
    exit 1
fi

MODELS=""

for MODEL in "$@"; do
    echo "Evaluating model $MODEL"

    # PIQA
    lm_eval --model hf --model_args pretrained=$MODEL --batch_size 8 --device cuda --trust_remote_code --tasks piqa --output_path "piqa_0shot"
    # SIQA
    lm_eval --model hf --model_args pretrained=$MODEL --batch_size 8 --device cuda --trust_remote_code --tasks social_iqa --output_path "siqa_0shot"
    # OpenBookQA
    lm_eval --model hf --model_args pretrained=$MODEL --batch_size 8 --device cuda --trust_remote_code --tasks openbookqa --output_path "openbookqa_0shot"
    # TriviaQA
    lm_eval --model hf --model_args pretrained=$MODEL --batch_size 8 --device cuda --trust_remote_code --tasks triviaqa --output_path "triviaqa_0shot"
    # TruthfulQA
    lm_eval --model hf --model_args pretrained=$MODEL --batch_size 8 --device cuda --trust_remote_code --tasks truthfulqa_mc1,truthfulqa_mc2 --output_path "truthfulqa_0shot"
    # MMLU
    lm_eval --model hf --model_args pretrained=$MODEL --batch_size 8 --device cuda --trust_remote_code --tasks mmlu --output_path "mmlu_5shot" --num_fewshot 5
    # WinoGrande
    lm_eval --model hf --model_args pretrained=$MODEL --batch_size 8 --device cuda --trust_remote_code --tasks winogrande --output_path "winogrande_5shot" --num_fewshot 5
    # Arc Challange
    lm_eval --model hf --model_args pretrained=$MODEL --batch_size 8 --device cuda --trust_remote_code --tasks arc_challenge --output_path "arc_challenge_25shot" --num_fewshot 25
    # HellaSwag
    lm_eval --model hf --model_args pretrained=$MODEL --batch_size 8 --device cuda --trust_remote_code --tasks hellaswag --output_path "hellaswag_10shot" --num_fewshot 10
    # GSM-8K
    lm_eval --model hf --model_args pretrained=$MODEL --batch_size 8 --device cuda --trust_remote_code --tasks gsm8k --output_path "gsm8k_5shot" --num_fewshot 5

    MODELS="$MODELS $MODEL"
done

python eval_results.py $MODELS