from pathlib import Path
import re
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt

from src.modeling.fusion_model import FusionModel

SEMANTIC_CLASSES = [
    "human face", "person", "vehicle", "animal", "text logo",
    "sports ball", "fireworks", "waterfall", "toy gun",
    "mountain", "building"
]

def parse_user(name: str) -> int:
    m = re.search(r"user(\d+)_", name)
    return int(m.group(1))

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("video_number", type=int, help="clip/video id (e.g., 18)")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=0.01)
    args = ap.parse_args()

    repo = Path(__file__).resolve().parents[1]
    combined_dir = repo / "data" / "combined"

    pt_paths = sorted(combined_dir.glob(f"user*_clip{args.video_number}_60hz.pt"))
    if not pt_paths:
        raise RuntimeError(
            f"No combined files for clip {args.video_number}. "
            f"Run: python -m scripts.run_dataset {args.video_number}"
        )

    # split by user (80/20)
    users = list({parse_user(p.name) for p in pt_paths})

    perm = torch.randperm(len(users))
    users = [users[i] for i in perm]

    split = max(1, int(0.8 * len(users)))
    train_users = set(users[:split])
    test_users = set(users[split:])

    train_paths = [p for p in pt_paths if parse_user(p.name) in train_users]
    test_paths  = [p for p in pt_paths if parse_user(p.name) in test_users]

    print(f"Total users: {len(users)}")
    print(f"Train users ({len(train_users)}): {sorted(train_users)}")
    print(f"Test users ({len(test_users)}): {sorted(test_users)}")

    def load_many(paths):
        m_all, s_all, u_all, y_all = [], [], [], []
        for p in paths:
            d = torch.load(p)
            m_all.append(d["motion_seq"])   # [N,30,2]
            s_all.append(d["semantic"])     # [N,11,4,6]
            u_all.append(d["user_id"])      # [N]
            y_all.append(d["target"])       # [N,2]
        return (torch.cat(m_all, 0),
                torch.cat(s_all, 0),
                torch.cat(u_all, 0).long(),
                torch.cat(y_all, 0))

    X_m_tr, X_s_tr, U_tr, Y_tr = load_many(train_paths)
    X_m_te, X_s_te, U_te, Y_te = load_many(test_paths)

    train_loader = DataLoader(
        TensorDataset(X_m_tr, X_s_tr, U_tr, Y_tr),
        batch_size=args.batch_size, shuffle=True
    )
    test_loader = DataLoader(
        TensorDataset(X_m_te, X_s_te, U_te, Y_te),
        batch_size=args.batch_size, shuffle=False
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    num_users = int(max(users)) + 1
    model = FusionModel(num_users=num_users, seq_len=30).to(device)

    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.MSELoss()

    train_losses, test_losses = [], []

    for epoch in range(args.epochs):
        model.train()
        tr = 0.0
        for m, s, u, y in train_loader:
            m = m.float().to(device)
            s = s.float().to(device)
            u = u.long().to(device)
            y = y.float().to(device)    
            if s.dim() == 5:
                s = s.mean(dim=1)
            opt.zero_grad()
            pred = model(m, s, u)
            loss = loss_fn(pred, y)
            loss.backward()
            opt.step()
            tr += loss.item()

        tr /= max(1, len(train_loader))
        train_losses.append(tr)

        model.eval()
        te = 0.0
        with torch.no_grad():
            for m, s, u, y in test_loader:
                m = m.float().to(device)
                s = s.float().to(device)
                u = u.long().to(device)
                y = y.float().to(device)
                te += loss_fn(model(m, s, u), y).item()
        te /= max(1, len(test_loader))
        test_losses.append(te)

        print(f"Epoch {epoch+1}/{args.epochs}: train={tr:.6f} test={te:.6f}")

    results = repo / "results" / f"clip{args.video_number}"
    results.mkdir(parents=True, exist_ok=True)

    # loss curve
    plt.figure()
    epochs = list(range(1, len(train_losses) + 1))
    plt.plot(epochs, train_losses, label="train", marker='o')
    plt.plot(epochs, test_losses, label="test", marker='o')

    # Annotate final values
    if train_losses:
        final_train = train_losses[-1]
        plt.scatter([epochs[-1]], [final_train], color='C0')
        plt.text(epochs[-1], final_train, f"  train: {final_train:.4f}",
                 verticalalignment='bottom', color='C0', fontsize=9,
                 bbox=dict(facecolor='white', alpha=0.6, edgecolor='none'))

    if test_losses:
        final_test = test_losses[-1]
        plt.scatter([epochs[-1]], [final_test], color='C1')
        plt.text(epochs[-1], final_test, f"  test: {final_test:.4f}",
                 verticalalignment='top', color='C1', fontsize=9,
                 bbox=dict(facecolor='white', alpha=0.6, edgecolor='none'))

    plt.xlabel("epoch")
    plt.ylabel("MSE")
    plt.legend()
    plt.title(f"FusionModel Loss (clip {args.video_number})")
    plt.tight_layout()
    plt.savefig(results / "loss_curve.png")
    plt.close()

    # save prefs + model
    prefs = model.user_preferences.weight.detach().cpu()
    torch.save(prefs, results / "user_preferences.pt")
    torch.save(model.state_dict(), results / f"{args.video_number}_prediction.pt")

    # === Plot BOTH test users on the SAME graph ===
    test_users_sorted = sorted(test_users)

    if len(test_users_sorted) == 0:
        print("No test users to plot.")
    else:
        # assume exactly 2 test users (80/20 split)
        u1 = test_users_sorted[0]
        u2 = test_users_sorted[1] if len(test_users_sorted) > 1 else None

        w1 = prefs[u1].numpy()
        w2 = prefs[u2].numpy() if u2 is not None else None

        import numpy as np
        x = np.arange(len(SEMANTIC_CLASSES))
        width = 0.35

        plt.figure(figsize=(12, 5))
        plt.bar(x - width/2, w1, width, label=f"User {u1}")
        if w2 is not None:
            plt.bar(x + width/2, w2, width, label=f"User {u2}")

        plt.xticks(x, SEMANTIC_CLASSES, rotation=45, ha="right")
        plt.ylabel("Preference weight")
        plt.title(f"User Preferences (Test Users) – Clip {args.video_number}")
        plt.legend()
        plt.tight_layout()
        plt.savefig(results / "test_users_prefs.png")
        plt.close()

        # === Generate predictions vs ground truth for each test user ===
        model.eval()
        with torch.no_grad():
            # Get predictions for all test data
            all_preds = []
            all_targets = []
            all_user_ids = []
            
            for m, s, u, y in test_loader:
                m = m.float().to(device)
                s = s.float().to(device)
                u = u.long().to(device)
                pred = model(m, s, u)
                all_preds.append(pred.cpu())
                all_targets.append(y)
                all_user_ids.append(u.cpu())
            
            all_preds = torch.cat(all_preds, dim=0).numpy()
            all_targets = torch.cat(all_targets, dim=0).numpy()
            all_user_ids = torch.cat(all_user_ids, dim=0).numpy()
        
        # Plot for each test user
        for test_user in test_users_sorted:
            mask = all_user_ids == test_user
            user_preds = all_preds[mask]
            user_targets = all_targets[mask]
            
            if len(user_preds) == 0:
                continue
            
            # Extract yaw and pitch
            pred_yaw = user_preds[:, 0]
            pred_pitch = user_preds[:, 1]
            true_yaw = user_targets[:, 0]
            true_pitch = user_targets[:, 1]
            
            # Create figure with two subplots
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
            
            # Yaw plot
            time_steps = np.arange(len(pred_yaw))
            ax1.plot(time_steps, true_yaw, 'b-', label='Ground Truth', alpha=0.7)
            ax1.plot(time_steps, pred_yaw, 'r--', label='Predicted', alpha=0.7)
            ax1.set_xlabel('Sample Index')
            ax1.set_ylabel('Yaw (degrees)')
            ax1.set_title(f'User {test_user} - Yaw Prediction vs Ground Truth')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Pitch plot
            ax2.plot(time_steps, true_pitch, 'b-', label='Ground Truth', alpha=0.7)
            ax2.plot(time_steps, pred_pitch, 'r--', label='Predicted', alpha=0.7)
            ax2.set_xlabel('Sample Index')
            ax2.set_ylabel('Pitch (degrees)')
            ax2.set_title(f'User {test_user} - Pitch Prediction vs Ground Truth')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(results / f"user{test_user}_predictions.png")
            plt.close()
            
            print(f"Saved prediction plot for User {test_user}")

        print("Saved to:", results)

if __name__ == "__main__":
    main()
