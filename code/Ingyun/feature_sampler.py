# train_dataloader에
# make_sampler()를 선언했고
# num_workers=8, sampler=make_sampler())#shuffle=args.shuffle
# 이것만 바꿔줬습니다.


    def train_dataloader(self):
        def make_sampler():
            train_data = pd.read_csv('data/train.csv')
            train_data['class'] = train_data['label'].apply(lambda x: int(x//1) if x!=5 else int(4))
            class_counts = train_data['class'].value_counts().to_list()
            # [3711, 1715, 1393, 1368, 1137]
            labels = train_data['class'].to_list()
            # [2, 4, 2, 3, 0, 2, 3, 0, 
            num_samples = sum(class_counts)
            class_weights = [num_samples/class_counts[i] for i in range(len(class_counts))]
            #[2.51, 5.436, 6.69, 6.81, 8.20]
            weights = [class_weights[labels[i]] for i in range(int(num_samples))]
            #[6.69, 8.20, 6.69, 6.81, 2.51, 6.69
            sampler = torch.utils.data.WeightedRandomSampler(torch.DoubleTensor(weights), int(num_samples))
            return sampler
        return torch.utils.data.DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=8, sampler=make_sampler())#shuffle=args.shuffle
