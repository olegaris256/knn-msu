from nearest_neighbors import KNNClassifier
import numpy as np

np.int = int


def kfold(n, n_folds):
    result = []
    fold_lenght = n // n_folds
    first_folds = n % n_folds
    for i in range(n_folds):
        if i < first_folds:
            ind1 = i * (fold_lenght + 1)
            ind2 = ind1 + (fold_lenght + 1)
        else:
            ind1 = i * fold_lenght + first_folds
            ind2 = ind1 + fold_lenght
        train_indices = np.concatenate((np.arange(0, ind1),
                                        np.arange(ind2, n)))
        val_indices = np.arange(ind1, ind2)
        result.append((train_indices, val_indices))
    return result


def knn_cross_val_score(X, y, k_list, score, cv=None, **kwargs):
    if cv is None:
        n = X.shape[0]
        n_folds = 5
        cv = kfold(n, n_folds)

    result = {k: np.zeros(len(cv)) for k in k_list}
    for fold_idx, (train_indices, val_indices) in enumerate(cv):
        X_train, y_train = X[train_indices], y[train_indices]
        X_val, y_val = X[val_indices], y[val_indices]
        for k in k_list:
            knn = KNNClassifier(k=k, **kwargs)
            knn.fit(X_train, y_train)
            y_pred = knn.predict(X_val)
            if score == 'accuracy':
                result[k][fold_idx] = (y_pred == y_val).sum() / len(y_pred)
    return result

