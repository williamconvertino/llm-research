short_test_sequences = [
  "There once was a fox who ",
  "There once was a fox who lived in the forest. He was very ",
  "There once was a fox who lived in the forest. He was very hungry and ",
  "There once was a princess who ",
  "There once was a princess who lived in a castle. She was very ",
  "There once was a princess who lived in a castle. She was very lonely and ",
]

long_test_sequences = [
  "There once was a fox who lived in the forest. He was very hungry and decided to go on a hunt. He found a rabbit and decided to chase it. The rabbit was very fast and the fox had to run very fast to catch it. He finally caught the rabbit and ate it. The fox was very happy and went back to his home in the forest. He was very tired and decided to take a nap. He woke up and saw a bear. The bear was very big and the fox was very scared. The bear started to chase the fox and the fox ran as fast as he could. The bear finally caught the fox and ",
  "There once was a princess who lived in a big castle. One day, the princess decided to go on an adventure. She packed her bags and left the castle. She walked through the forest and saw a dragon. The dragon was very big and the princess was very scared. The dragon started to chase the princess and the princess ran as fast as she could. The dragon finally caught the princess and ",
]

def evaluate_model_generation(model, tokenizer, val_dataset):
  
  for sequence in (short_test_sequences + long_test_sequences):
    print(f"Input: {sequence}")
    for i in range(5):
      print(f"Generated {i+1}: {model.generate(sequence, n=10, tokenizer=tokenizer)}")