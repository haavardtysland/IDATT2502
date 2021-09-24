import torch
import torch.nn as nn


class LongShortTermMemoryModel(nn.Module):
    def __init__(self, encoding_size, label_size):
        super(LongShortTermMemoryModel, self).__init__()
        self.lstm = nn.LSTM(char_encoding_size, 128)  # 128 is the state size
        self.fc1 = nn.Linear(128, emoji_encoding_size)

    def reset(self, batch_size=1):
        zero_state = torch.zeros(1, batch_size, 128)
        self.hidden_state = zero_state
        self.cell_state = zero_state

    def logits(self, x):  # x shape: (sequence length, batch size, encoding size)
        out, (self.hidden_state, self.cell_state) = self.lstm(
            x, (self.hidden_state, self.cell_state))
        return self.fc1(out.reshape(-1, 128))

    def f(self, x):  # x shape: (sequence length, batch size, encoding size)
        return torch.softmax(self.logits(x), dim=1)

    def loss(self, x, y):  # x shape: (sequence length, batch size, encoding size), y shape: (sequence length, encoding size)
        return nn.functional.cross_entropy(self.logits(x), y.argmax(1))




char_encodings = [
        [1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],  # ' ' 0
        [0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],  # 'h' 1
        [0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],  # 'a' 2
        [0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0.],  # 't' 3
        [0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0.],  # 'r' 4
        [0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0.],  # 'c' 5
        [0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0.],  # 'f' 6
        [0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0.],  # 'l' 7
        [0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0.],  # 'm' 8
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0.],  # 'p' 9
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0.],  # 's' 10
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0.],  # 'o' 11
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1.]   # 'n' 12
    ]
char_encoding_size = len(char_encodings)
index_to_char = [' ', 'h', 'a', 't', 'r','c', 'f', 'l', 'm', 'p', 's', 'o', 'n']

emojis = {
    'hat': '\U0001F3A9',
    'cat': '\U0001F408',
    'rat': '\U0001F400',
    'flat': '\U0001F3E2',
    'matt': '\U0001F468',
    'cap': '\U0001F9E2',
    'son': '\U0001F466'
}

emoji_enc = [
    [1., 0., 0., 0., 0., 0., 0.],  # 'hat' 0
    [0., 1., 0., 0., 0., 0., 0.],  # 'rat' 1
    [0., 0., 1., 0., 0., 0., 0.],  # 'cat' 2
    [0., 0., 0., 1., 0., 0., 0.],  # 'flat' 3
    [0., 0., 0., 0., 1., 0., 0.],  # 'matt' 4
    [0., 0., 0., 0., 0., 1., 0.],  # 'cap' 5
    [0., 0., 0., 0., 0., 0., 1.]   # 'son' 6
]

emoji_encoding_size = len(emoji_enc)
index_to_emoji = [emojis['hat'], emojis['rat'], emojis['cat'],
                      emojis['flat'], emojis['matt'], emojis['cap'], emojis['son']]

#Legger mellomrom på slutten av ordene med 3 bokstaver
x_train = torch.tensor([
        [[char_encodings[1]], [char_encodings[2]], [char_encodings[3]], [char_encodings[0]]], #hat
        [[char_encodings[4]], [char_encodings[2]], [char_encodings[3]], [char_encodings[0]]], #rat
        [[char_encodings[5]], [char_encodings[2]], [char_encodings[3]], [char_encodings[0]]], #cat
        [[char_encodings[6]], [char_encodings[7]], [char_encodings[2]], [char_encodings[3]]], #flat
        [[char_encodings[8]], [char_encodings[2]], [char_encodings[3]], [char_encodings[3]]], #matt
        [[char_encodings[5]], [char_encodings[2]], [char_encodings[9]], [char_encodings[0]]], #cap
        [[char_encodings[10]], [char_encodings[11]], [char_encodings[12]], [char_encodings[0]]]]) #son

y_train = torch.tensor([
    [emoji_enc[0], emoji_enc[0], emoji_enc[0], emoji_enc[0]],
    [emoji_enc[1], emoji_enc[1], emoji_enc[1], emoji_enc[1]],
    [emoji_enc[2], emoji_enc[2], emoji_enc[2], emoji_enc[2]],
    [emoji_enc[3], emoji_enc[3], emoji_enc[3], emoji_enc[3]],
    [emoji_enc[4], emoji_enc[4], emoji_enc[4], emoji_enc[4]],
    [emoji_enc[5], emoji_enc[5], emoji_enc[5], emoji_enc[5]],
    [emoji_enc[6], emoji_enc[6], emoji_enc[6], emoji_enc[6]]])

model = LongShortTermMemoryModel(char_encoding_size, emoji_encoding_size)

def generate(string):
    model.reset()
    for i in range(len(string)):
        char_index = index_to_char.index(string[i])
        y = model.f(torch.tensor([[char_encodings[char_index]]]))
        if i == len(string) - 1:
            print(index_to_emoji[y.argmax(1)])

optimizer = torch.optim.RMSprop(model.parameters(), 0.001)  # 0.001
for epoch in range(500):
    for i in range(x_train.size()[0]):
        model.reset()
        model.loss(x_train[i], y_train[i]).backward()
        optimizer.step()
        optimizer.zero_grad()

    if epoch % 10 == 9:
        text = ''

model.reset()
print("Input: rat")
for i in range(3):
    word = "rat"
    index = index_to_char.index(word[i])
    y = model.f(torch.tensor([[char_encodings[index]]]))
print(index_to_emoji[y.argmax(1)])

model.reset()
print("Input: cat")
for i in range(3):
    word = "hat"
    index = index_to_char.index(word[i])
    y = model.f(torch.tensor([[char_encodings[index]]]))
print(index_to_emoji[y.argmax(1)])

    