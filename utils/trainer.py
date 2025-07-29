import torch
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix, f1_score, roc_auc_score

@torch.no_grad()
def evaluate(model, data):
    model.eval()
    logits = model(data)
    pred = logits.argmax(dim=1)

    results = {}

    for name, mask in zip(["train", "val", "test"], [data.train_mask, data.val_mask, data.test_mask]):
        true_labels = data.y[mask].cpu().numpy()
        pred_labels = pred[mask].cpu().numpy()

        f1 = f1_score(true_labels, pred_labels, average="macro")
        cm = confusion_matrix(true_labels, pred_labels)
        try:
            roc_auc = roc_auc_score(
                torch.nn.functional.one_hot(torch.tensor(true_labels), num_classes=logits.size(1)),
                torch.softmax(logits[mask], dim=1).cpu().numpy(),
                multi_class="ovr"
            )
        except ValueError:
            roc_auc = None  # Not computable for certain edge cases

        acc = (pred_labels == true_labels).sum() / len(true_labels)

        results[name] = {
            "acc": acc,
            "f1": f1,
            "roc_auc": roc_auc,
            "confusion_matrix": cm
        }

    return results

def train_model(model, data, optimizer, epochs=100):
    train_accs, val_accs, test_accs = [], [], []
    f1_scores, roc_aucs = [], []

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        out = model(data)
        loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()

        results = evaluate(model, data)
        train_accs.append(results["train"]["acc"])
        val_accs.append(results["val"]["acc"])
        test_accs.append(results["test"]["acc"])
        f1_scores.append(results["test"]["f1"])
        roc_aucs.append(results["test"]["roc_auc"])

    return train_accs, val_accs, test_accs, f1_scores, roc_aucs, results["test"]["confusion_matrix"]