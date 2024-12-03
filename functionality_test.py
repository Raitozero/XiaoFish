import subprocess

def test_functionality(input_commands, reference_commands, generated_commands):
    def execute_command(command):
        try:
            result = subprocess.run(
                command,
                shell=True,
                text=True,
                capture_output=True,
                timeout=5  # Avoid infinite loops
            )
            return result.stdout.strip(), result.stderr.strip()
        except subprocess.TimeoutExpired:
            return "", "Command timed out"
        except Exception as e:
            return "", f"Error: {str(e)}"

    results = []
    for i, (ref_cmd, gen_cmd) in enumerate(zip(reference_commands, generated_commands)):
        # Compare commands case-insensitively
        if ref_cmd.strip().lower() == gen_cmd.strip().lower():
            results.append({
                "input": input_commands[i],
                "reference": ref_cmd,
                "generated": gen_cmd,
                "ref_output": "",  # Skip execution
                "gen_output": "",
                "ref_error": "",
                "gen_error": "",
                "correct": True,
            })
            continue

        # Execute commands if they differ
        ref_out, ref_err = execute_command(ref_cmd)
        gen_out, gen_err = execute_command(gen_cmd)
        is_correct = (ref_out == gen_out) and (ref_err == gen_err)
        results.append({
            "input": input_commands[i],
            "reference": ref_cmd,
            "generated": gen_cmd,
            "ref_output": ref_out,
            "gen_output": gen_out,
            "ref_error": ref_err,
            "gen_error": gen_err,
            "correct": is_correct,
        })
    return results


def get_accuracy(results):
    num_true = 0
    count = 0
    for res in results:
        count += 1
        if res["correct"] == True:
            num_true += 1
    return [count, num_true]



# # Example Usage
# input_commands = ["List all files including hidden ones"]
# reference_commands = ["ls -l -a"]
# generated_commands = ["ls -a -l"]

# results = test_functionality(input_commands, reference_commands, generated_commands)

# for res in results:
#     print(f"Input: {res['input']}")
#     print(f"Reference: {res['reference']}")
#     print(f"Generated: {res['generated']}")
#     print(f"Correct: {res['correct']}\n")
