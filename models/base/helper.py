# import new Network name here and add in model_class args
from .Network import MYNET
from utils import *
from tqdm import tqdm
import torch.nn.functional as F
from models.loss_function import *

def base_train(model, trainloader, optimizer, scheduler, epoch, args, criterion):

    torch.manual_seed(time.time())

    tl = Averager()
    ta = Averager()

    model = model.train()
    # standard classification for pretrain
    tqdm_gen = tqdm(trainloader)
    beta = 0.5

    for i, batch in enumerate(tqdm_gen, 1):
        data, train_label = [_.cuda() for _ in batch]

        logits = model(data)
        feats = model.module.encode(data)
        logits = logits[:, :args.base_class]

        if args.extra_class:
            logits = logits[:, :args.base_class+12]

        loss = F.cross_entropy(logits, train_label)
        total_loss = loss

        # mixup
        if args.mix>0:

            index = torch.randperm(data.size(0)).cuda()
            pre_emb1=model.module.pre_encode(data)
            mixed_data=beta*pre_emb1+(1-beta)*pre_emb1[index]
            mixed_logits=model.module.post_encode(mixed_data)
            mixed_feats=model.module.post_encode(mixed_data, feat=True)

            newys=train_label[index]
            idx_chosen=newys!=train_label
            mixed_logits=mixed_logits[idx_chosen]
            mixed_feats=mixed_feats[idx_chosen]

            pseudo_label1 = torch.argmax(mixed_logits[:,args.base_class:], dim=-1) + args.base_class # new class label
            mix_loss = F.cross_entropy(mixed_logits, pseudo_label1)

            total_loss = loss + args.mix*mix_loss

        acc = count_acc(logits, train_label)
        lrc = scheduler.get_last_lr()[0]
        tqdm_gen.set_description(
            'Session 0, epo {}, lrc={:.4f},total loss={:.4f} acc={:.4f}'.format(epoch, lrc, total_loss.item(), acc))
        tl.add(total_loss.item())
        ta.add(acc)

        optimizer.zero_grad()
        total_loss.backward()


        optimizer.step()

    tl = tl.item()
    ta = ta.item()
    return tl, ta


def replace_base_fc(trainset, transform, model, args):
    # replace fc.weight with the embedding average of train data
    model = model.eval()

    # if args.dataset == 'auo':
    #     trainset = args.Dataset.AUO(root=args.dataroot, train=False, index=np.arange(args.base_class), proto=True)
    trainloader = torch.utils.data.DataLoader(dataset=trainset, batch_size=128,
                                              num_workers=args.num_workers, pin_memory=True, shuffle=False)
    trainloader.dataset.transform = transform
    embedding_list = []
    label_list = []
    # data_list=[]
    with torch.no_grad():
        for i, batch in enumerate(trainloader):
            data, label = [_.cuda() for _ in batch]
            model.module.mode = 'encoder'
            embedding = model(data)

            embedding_list.append(embedding.cpu())
            label_list.append(label.cpu())
    embedding_list = torch.cat(embedding_list, dim=0)
    label_list = torch.cat(label_list, dim=0)

    proto_list = []

    for class_index in range(args.base_class):
        data_index = (label_list == class_index).nonzero()
        embedding_this = embedding_list[data_index.squeeze(-1)]
        embedding_this = embedding_this.mean(0)
        proto_list.append(embedding_this)

    proto_list = torch.stack(proto_list, dim=0)

    model.module.fc.weight.data[:args.base_class] = proto_list

    return model


def test(model, testloader, epoch, args, session, criterion):
    label_list = {}
    label_list['true_list'] = []
    label_list['pred_list'] = []
    test_class = args.base_class + session * args.way
    model = model.eval()
    vl = Averager()
    va = Averager()
    with torch.no_grad():
        tqdm_gen = tqdm(testloader)
        for i, batch in enumerate(tqdm_gen, 1):
            data, test_label = [_.cuda() for _ in batch]
            logits = model(data)
            feats = model.module.encode(data)
            logits = logits[:, :test_class]
            # _, mlogits, _ = criterion[0](feats, test_label)
            loss = F.cross_entropy(logits, test_label)
            acc = count_acc(logits, test_label)

            vl.add(loss.item())
            va.add(acc)

            p = torch.argmax(logits, dim=1).cpu().numpy()
            t = test_label.cpu().numpy()
            label_list['pred_list'].extend(p)
            label_list['true_list'].extend(t)

        vl = vl.item()
        va = va.item()
    
    print('epo {}, test, loss={:.4f} acc={:.4f}'.format(epoch, vl, va))

    return vl, va, label_list
