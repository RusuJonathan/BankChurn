import pandas as pd
from src.models.model import Model
from src.utils.utils import rank_average
from src.data.data_loader import (
    load_yaml,
    load_data,
    train_path,
    test_path,
    model_config_path,
    best_hyperparams_path,
    target,
)

df     = load_data(train_path)
x_test = load_data(test_path)
config = load_yaml(model_config_path)
best_params = load_yaml(best_hyperparams_path)

x_train = df.drop(columns=[target])
y_train = df[target]

model = Model(config=config, n_trials=40)

model.optimize_hyperparameters(x_train, y_train)
#model.load_hyperparameters(best_params)

model.fit(x_train, y_train)

stacking_proba = model.predict_proba(x_test)[:, 1]

#Average stacking + raw base model probas
individual_probas = []
for name, pipeline in model.estimators:
    pipeline.fit(x_train, y_train)
    proba = pipeline.predict_proba(x_test)[:, 1]
    individual_probas.append(proba)

# Weight stacking more heavily than individual models
all_probas = [stacking_proba] * 3 + individual_probas
blended_proba = rank_average(all_probas)

#blended_proba = stacking_proba

output = pd.DataFrame({
    "id":     x_test["id"],
    "Exited": blended_proba,
})
output.to_csv("submission.csv", index=False)