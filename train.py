import torch
import torch.nn as nn
def train_on_policy(model, dataloader, epochs, clip, lr):
    pass

def train_taid(teacher_model, student_model, tokenizer, dataloader, epochs, lr, beta, alpha):  
    epsilon = 1e-8
    t_linear = torch.linspace(0, 1, steps=epochs * len(dataloader))
    student_model.train()
    teacher_model.eval()

    def calculate_taid_loss(teacher_outputs, student_outputs, t):
        detached_student_outputs = student_outputs.detach().clone()
        p_t = nn.functional.softmax((1 - t) * detached_student_outputs + t * teacher_outputs, dim=-1)
        obj = nn.functional.kl_div(student_outputs, p_t, reduction="mean")
        return obj

    def calculate_taid_t(current_loss, previous_loss, previous_t, previous_m, step):
        delta = (previous_loss - current_loss) / (previous_loss + epsilon)
        m = beta * previous_m + (1 - beta) * delta
        delta_t = alpha * torch.nn.functional.sigmoid(m) * (1 - previous_t)
        t_n1 = min(torch.tensor(1), max(t_linear[step], previous_t + delta_t))
        return t_n1, m
    
    t = torch.tensor(0)
    optimizer = torch.optim.SGD(student_model.parameters(), lr=lr, momentum=0.9)
    for epoch in epochs:
        for seqs in dataloader:
            optimizer.zero_grad()
            inputs = tokenizer(seqs, return_tensors="pt", padding=True, truncation=True)
            input_ids = inputs["input_ids"]          # shape: [batch_size, seq_len]
            attention_mask = inputs["attention_mask"]

            labels = input_ids.clone()
            labels[labels == tokenizer.pad_token_id] = -100

            student_outputs = student_model(input_ids, attention_mask=attention_mask, labels=labels)
            student_logits = student_outputs.logits

            teacher_outputs = teacher_model(input_ids, attention_mask=attention_mask, labels=labels)
            teacher_logits = teacher_outputs.logits
            
            loss = calculate_taid_loss(teacher_logits, student_logits, t)
            loss.backward()
            optimizer.step()
            if epoch == 0 and step == 0:
                previous_loss = loss.item()
                previous_t = t
                previous_m = torch.tensor(0.0)
            else:
                t, m = calculate_taid_t(loss.item(), previous_loss, previous_t, previous_m, step)
                previous_loss = loss.item()
                previous_t = t
                previous_m = m
            step += 1
    return student_model
            







    
