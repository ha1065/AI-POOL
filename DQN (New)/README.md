
#Execute all this within root of DQN folder :


# (Optional) Create & activate virtual environment
$ virtualenv .venv
$ source .venv/bin/activate

#Look at requirements.txt , if all requirements are satisfied ok to run. Otherwise install package :
$ python -m pip install -r requirements.txt

# Run game
$ python -m src.game.main

# Run training, ALGO = 'dqn' , BALLS = 2 or more , --visualize optional if want to veiw training epochs
$ python -m src.model.train [--balls BALLS] [--algo ALGO] [--visualize] output_model

# Run evaluation, ALGO = dqn , BALLS = 2 or more , MODEL = name of output_model you used , --visualize optional if you want to view testing epochs
$ python -m src.model.eval [--model MODEL] [--balls BALLS] [--algo ALGO] [--visualize]
```

## Visualizing Average train rewards

- Visualize average rewards over epochs
    - `$ python -m src.utils.training_rewards_dqn_vis dqn-log.txt OUTPUT_FILE`

## Test Rewards and training loss automatically visualized after end of test and training epochs respectively. Will need to save plots generated locally.


