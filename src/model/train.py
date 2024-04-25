from src.config import *
from src.loaddata.data_loader import get_loader
from src.model.prototype_loss import prototypical_loss


def trainNet(opt):
    train_loader = get_loader(batch_size=opt.batch_size, num_workers=16, path=TRAIN_PATH)
    test_loader = get_loader(batch_size=opt.batch_size, num_workers=16, path=TEST_PATH)
    model = LighterAlexNet(num_classes=opt.n_classes).cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr,
                                 betas=(0.5, 0.999), eps=1e-08,
                                 weight_decay=2e-05)
    scheduler = ReduceLROnPlateau(optimizer, factor=0.2, patience=8, mode='min',
                                  min_lr=1e-05)

    loss = torch.nn.CrossEntropyLoss()

    for epoch in range(opt.dl_n_epochs):
        ground_truth = []
        y_pred = []
        model.train()
        print('\n==================== Epoch: {} ==================\n'.format(epoch))
        for i, data in enumerate(train_loader, 0):
            _, images, ID, labels = data
            images, labels = Variable(images).cuda(), Variable(labels).cuda()
            ground_truth.extend(Variable(labels).data.cpu().numpy())
            optimizer.zero_grad()
            outputs = model(images)
            loss_size = loss(outputs, labels)
            # loss_size, _ = prototypical_loss(outputs, target=labels, n_support=10)
            _, predicted = torch.max(outputs.data, 1)
            y_pred.extend(Variable(predicted).data.cpu().numpy())
            loss_size.backward()
            optimizer.step()
        print(classification_report(y_pred=y_pred, y_true=ground_truth))

        if epoch == opt.dl_n_epochs-1:
            torch.save(model, os.path.join('saved_pkls', opt.virus_type+'.pkl'))

        print('\n================= Validating/Testing ==================\n')
        with torch.no_grad():
            model.eval()

            testing_ground_truth = []
            testing_y_pred = []

            for data in test_loader:
                _, images, ID, labels = data
                images, labels = Variable(images).cuda(), Variable(labels).cuda()
                testing_ground_truth.extend(Variable(labels).data.cpu().numpy())

                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                testing_y_pred.extend(Variable(predicted).data.cpu().numpy())

            val_acc = accuracy_score(y_pred=testing_y_pred, y_true=testing_ground_truth)

            for param_group in optimizer.param_groups:
                old_lr = param_group['lr']
                print(param_group['lr'])

            scheduler.step(val_acc)
            for param_group in optimizer.param_groups:
                new_lr = param_group['lr']
                if new_lr < old_lr:
                    print("change learning rate at epoch {} from {} to {}".format(epoch, old_lr, new_lr))

            print(classification_report(y_pred=testing_y_pred, y_true=testing_ground_truth))
            model.train()
