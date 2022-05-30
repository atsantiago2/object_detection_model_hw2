import torch
import torchaudio, torchvision
import os
import matplotlib.pyplot as plt 
import librosa
import argparse
import numpy as np
import wandb
from pytorch_lightning import LightningModule, Trainer, LightningDataModule, Callback
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from torchmetrics.functional import accuracy
from torchvision.transforms import ToTensor
from torchaudio.datasets import SPEECHCOMMANDS
from torchaudio.datasets.speechcommands import load_speechcommands_item



class SilenceDataset(SPEECHCOMMANDS):
    def __init__(self, root):
        super(SilenceDataset, self).__init__(root, subset='training')
        self.len = len(self._walker) // 35
        path = os.path.join(self._path, torchaudio.datasets.speechcommands.EXCEPT_FOLDER)
        self.paths = [os.path.join(path, p) for p in os.listdir(path) if p.endswith('.wav')]

    def __getitem__(self, index):
        index = np.random.randint(0, len(self.paths))
        filepath = self.paths[index]
        waveform, sample_rate = torchaudio.load(filepath)
        return waveform, sample_rate, "silence", 0, 0

    def __len__(self):
        return self.len

class UnknownDataset(SPEECHCOMMANDS):
    def __init__(self, root):
        super(UnknownDataset, self).__init__(root, subset='training')
        self.len = len(self._walker) // 35

    def __getitem__(self, index):
        index = np.random.randint(0, len(self._walker))
        fileid = self._walker[index]
        waveform, sample_rate, _, speaker_id, utterance_number = load_speechcommands_item(fileid, self._path)
        return waveform, sample_rate, "unknown", speaker_id, utterance_number

    def __len__(self):
        return self.len







class KWSDataModule(LightningDataModule):
    def __init__(self, path, batch_size=128, num_workers=0, n_fft=512, 
                 n_mels=128, win_length=None, hop_length=256, class_dict={}, 
                 **kwargs):
        print('_'*20, 'kws init')
        super().__init__(**kwargs)
        self.path = path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.n_fft = n_fft
        self.n_mels = n_mels
        self.win_length = win_length
        self.hop_length = hop_length
        self.class_dict = class_dict

    def prepare_data(self):
        print('_'*20, 'kws prep data')
        self.train_dataset = torchaudio.datasets.SPEECHCOMMANDS(self.path,
                                                                download=True,
                                                                subset='training')

        silence_dataset = SilenceDataset(self.path)
        unknown_dataset = UnknownDataset(self.path)
        self.train_dataset = torch.utils.data.ConcatDataset([self.train_dataset, silence_dataset, unknown_dataset])
                                                                
        self.val_dataset = torchaudio.datasets.SPEECHCOMMANDS(self.path,
                                                              download=True,
                                                              subset='validation')
        self.test_dataset = torchaudio.datasets.SPEECHCOMMANDS(self.path,
                                                               download=True,
                                                               subset='testing')                                                    
        _, sample_rate, _, _, _ = self.train_dataset[0]
        self.sample_rate = sample_rate
        self.transform = torchaudio.transforms.MelSpectrogram(sample_rate=sample_rate,
                                                              n_fft=self.n_fft,
                                                              win_length=self.win_length,
                                                              hop_length=self.hop_length,
                                                              n_mels=self.n_mels,
                                                              power=2.0)

    def setup(self, stage=None):
        print('_'*20, 'setup ')
        self.prepare_data()

    def train_dataloader(self):
        print('_'*20, 'kws train dataloader')
        return torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            pin_memory=True,
            collate_fn=self.collate_fn
        )

    def val_dataloader(self):
        print('_'*20, 'kws val dataloader')
        return torch.utils.data.DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=True,
            collate_fn=self.collate_fn
        )
    
    def test_dataloader(self):
        print('_'*20, 'kws test dataloader')
        return torch.utils.data.DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            pin_memory=True,
            collate_fn=self.collate_fn
        )

    def collate_fn(self, batch):
        print('_'*20, 'kws collate fn')
        mels = []
        labels = []
        wavs = []
        for sample in batch:
            print('\nsample:', sample)
            waveform, sample_rate, label, speaker_id, utterance_number = sample
            # ensure that all waveforms are 1sec in length; if not pad with zeros
            if waveform.shape[-1] < sample_rate:
                waveform = torch.cat([waveform, torch.zeros((1, sample_rate - waveform.shape[-1]))], dim=-1)
            elif waveform.shape[-1] > sample_rate:
                waveform = waveform[:,:sample_rate]

            # mel from power to db
            mel1 = ToTensor()(librosa.power_to_db(self.transform(waveform).squeeze().numpy(), ref=np.max))
            print('\nMEL 1 Shape:', mel1.shape)
            print('\nMEL 1:', mel1)

            label1 = torch.tensor(self.class_dict[label])
            print('\nlabel:', label1.shape)

            print('\nwavform:', waveform.shape)
            mels.append(mel1)
            labels.append(label1)
            wavs.append(waveform)

        mels = torch.stack(mels)
        labels = torch.stack(labels)
        wavs = torch.stack(wavs)
    
        # print('mels', mels.shape)
        # print('labels', labels.shape)
        # print('wavs', wavs.shape)
        input()

        return mels, labels, wavs





#  ______________________________________________________________________________


class LitTransformer(LightningModule):
    def __init__(self, num_classes=10, lr=0.001, max_epochs=30, depth=12, embed_dim=64,
                 head=4, patch_dim=192, seqlen=16, **kwargs):
        super().__init__()
        print('_'*20, 'init')
        self.save_hyperparameters()
        self.encoder = Transformer(dim=embed_dim, num_heads=head, num_blocks=depth, mlp_ratio=4.,
                                   qkv_bias=False, act_layer=nn.GELU, norm_layer=nn.LayerNorm)
        self.embed = torch.nn.Linear(patch_dim, embed_dim)

        self.fc = nn.Linear(seqlen * embed_dim, num_classes)
        self.loss = torch.nn.CrossEntropyLoss()
        
        self.reset_parameters()


    def reset_parameters(self):
        print('_'*20, 'reset param')
        init_weights_vit_timm(self)
    

    def forward(self, x):
        print('_'*20, 'fwd')
        # Linear projection
        x = self.embed(x)
            
        # Encoder
        x = self.encoder(x)
        x = x.flatten(start_dim=1)

        # Classification head
        x = self.fc(x)
        return x
    
    def configure_optimizers(self):
        print('_'*20, 'config optimizers')
        optimizer = Adam(self.parameters(), lr=self.hparams.lr)
        # this decays the learning rate to 0 after max_epochs using cosine annealing
        scheduler = CosineAnnealingLR(optimizer, T_max=self.hparams.max_epochs)
        return [optimizer], [scheduler]

    def training_step(self, batch, batch_idx):
        print('_'*20, 'steps')
        x, y = batch
        y_hat = self(x)
        loss = self.loss(y_hat, y)
        return loss
    

    def test_step(self, batch, batch_idx):
        # print('_'*20, 'test step')
        x, y = batch
        y_hat = self(x)
        loss = self.loss(y_hat, y)
        acc = accuracy(y_hat, y)
        return {"y_hat": y_hat, "test_loss": loss, "test_acc": acc}

    def test_epoch_end(self, outputs):
        # print('_'*20, 'test ep end')
        avg_loss = torch.stack([x["test_loss"] for x in outputs]).mean()
        avg_acc = torch.stack([x["test_acc"] for x in outputs]).mean()
        self.log("test_loss", avg_loss, on_epoch=True, prog_bar=True)
        self.log("test_acc", avg_acc*100., on_epoch=True, prog_bar=True)

    def validation_step(self, batch, batch_idx):
        # print('_'*20, 'valid step')
        return self.test_step(batch, batch_idx)

    def validation_epoch_end(self, outputs):
        # print('_'*20, 'vald epoc end')
        return self.test_epoch_end(outputs)


# a lightning data module for cifar 10 dataset
class LitCifar10(LightningDataModule):
    def __init__(self, batch_size=32, num_workers=32, patch_num=4, **kwargs):
        print('_'*20, 'datamodule init')
        super().__init__()
        self.batch_size = batch_size
        self.patch_num = patch_num
        self.num_workers = num_workers

    def prepare_data(self):
        print('_'*20, 'datamodule prepare data')
        self.train_set = CIFAR10(root='./data', train=True,
                                 download=True, transform=torchvision.transforms.ToTensor())
        self.test_set = CIFAR10(root='./data', train=False,
                                download=True, transform=torchvision.transforms.ToTensor())

    def collate_fn(self, batch):
        print('_'*20, 'datamodule collate fn')
        x, y = zip(*batch)
        x = torch.stack(x, dim=0)
        y = torch.LongTensor(y)
        x = rearrange(x, 'b c (p1 h) (p2 w) -> b (p1 p2) (c h w)', p1=self.patch_num, p2=self.patch_num)
        return x, y

    def train_dataloader(self):
        print('_'*20, 'datamodule train dataloader')
        return torch.utils.data.DataLoader(self.train_set, batch_size=self.batch_size, 
                                        shuffle=True, collate_fn=self.collate_fn,
                                        num_workers=self.num_workers)

    def test_dataloader(self):
        print('_'*20, 'datamodule test dataload')
        return torch.utils.data.DataLoader(self.test_set, batch_size=self.batch_size, 
                                        shuffle=False, collate_fn=self.collate_fn,
                                        num_workers=self.num_workers)

    def val_dataloader(self):
        print('_'*20, 'datamodule valdataloader')
        return self.test_dataloader()

#  ______________________________________________________________________________











def get_args():
    parser = argparse.ArgumentParser(description='PyTorch Transformer')
    parser.add_argument('--depth', type=int, default=12, help='depth')
    parser.add_argument('--embed_dim', type=int, default=64, help='embedding dimension')
    parser.add_argument('--num_heads', type=int, default=4, help='num_heads')

    parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--max-epochs', type=int, default=30, metavar='N',
                        help='number of epochs to train (default: 30)')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 0.001)')
    parser.add_argument('--patch_num', type=int, default=8, help='patch_num')
    parser.add_argument('--kernel_size', type=int, default=3, help='kernel size')

    parser.add_argument('--accelerator', default='gpu', type=str, metavar='N')
    parser.add_argument('--devices', default=1, type=int, metavar='N')
    parser.add_argument('--dataset', default='cifar10', type=str, metavar='N')
    parser.add_argument('--num_workers', default=4, type=int, metavar='N')


    # where dataset will be stored
    parser.add_argument("--path", type=str, default="data/speech_commands/")
    # 35 keywords + silence + unknown
    parser.add_argument("--num-classes", type=int, default=37)

    # mel spectrogram parameters
    parser.add_argument("--n-fft", type=int, default=1024)
    parser.add_argument("--n-mels", type=int, default=128)
    parser.add_argument("--win-length", type=int, default=None)
    parser.add_argument("--hop-length", type=int, default=512)


    # 16-bit fp model to reduce the size
    parser.add_argument("--precision", default=16)
    # parser.add_argument("--accelerator", default='gpu')
    # parser.add_argument("--devices", default=1)
    # parser.add_argument("--num-workers", type=int, default=96)

    parser.add_argument("--no-wandb", default=True, action='store_true')

    
    args = parser.parse_args("")
    return args

class WandbCallback(Callback):

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        # log 10 sample audio predictions from the first batch
        if batch_idx == 0:
            n = 10
            mels, labels, wavs = batch
            preds = outputs["preds"]
            preds = torch.argmax(preds, dim=1)

            labels = labels.cpu().numpy()
            preds = preds.cpu().numpy()
            
            wavs = torch.squeeze(wavs, dim=1)
            wavs = [ (wav.cpu().numpy()*32768.0).astype("int16") for wav in wavs]
            
            sample_rate = pl_module.hparams.sample_rate
            idx_to_class = pl_module.hparams.idx_to_class
            
            # log audio samples and predictions as a W&B Table
            columns = ['audio', 'mel', 'ground truth', 'prediction']
            data = [[wandb.Audio(wav, sample_rate=sample_rate), wandb.Image(mel), idx_to_class[label], idx_to_class[pred]] for wav, mel, label, pred in list(
                zip(wavs[:n], mels[:n], labels[:n], preds[:n]))]
            wandb_logger.log_table(
                key='ResNet18 on KWS using PyTorch Lightning',
                columns=columns,
                data=data)





CLASSES = ['silence', 'unknown', 'backward', 'bed', 'bird', 'cat', 'dog', 'down', 'eight', 'five', 'follow',
            'forward', 'four', 'go', 'happy', 'house', 'learn', 'left', 'marvin', 'nine', 'no',
            'off', 'on', 'one', 'right', 'seven', 'sheila', 'six', 'stop', 'three',
            'tree', 'two', 'up', 'visual', 'wow', 'yes', 'zero']
# make a dictionary from CLASSES to integers
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}





if __name__ == "__main__":
    # wandb is a great way to debug and visualize this model
    wandb_logger = WandbLogger(project="pl-kws")
    args = get_args()

    if not os.path.exists(args.path):
        os.makedirs(args.path, exist_ok=True)


    # model = KWSModel(num_classes=args.num_classes, epochs=args.max_epochs, lr=args.lr)

    model = LitTransformer(num_classes=10, lr=args.lr, epochs=args.max_epochs, 
                           depth=args.depth, embed_dim=args.embed_dim, head=args.num_heads,
                           patch_dim=patch_dim, seqlen=seqlen,)
    print(model)






    # trainer = Trainer(accelerator=args.accelerator, devices=args.devices,
    #                   max_epochs=args.max_epochs, precision=16 if args.accelerator == 'gpu' else 32,)
    # trainer.fit(model, datamodule=datamodule)
    wandb.finish()