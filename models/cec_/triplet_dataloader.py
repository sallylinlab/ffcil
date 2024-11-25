            data, true_label = [_.cuda() for _ in batch]
            
            true_label = torch.cat((true_label, label), dim=0)[k:]
            model.module.mode = 'encoder'


            pseudo_index = []
            masks = []
            train_proto = list(np.arange(args.episode_way))
            while len(pseudo_index)<args.low_way:
                t = sorted(random.sample(train_proto, 2))
                if t not in pseudo_index:
                    pseudo_index.append(t)
                    

            k = args.episode_way * args.episode_shot
            # proto = 00000 11111 22222 33333, way=4 shot=5
            proto, query = data[:k], data[k:]
            proto_tmp, query_tmp = self.get_rotate_pseudo_classes(proto, query)
            proto_tmp = model(proto_tmp)
            query_tmp = model(query_tmp)

            k2 = args.episode_way * args.episode_query
            proto = data[:k]
            query = data[k:]

            proto = model(proto)
            query = model(query)
            labels = []
            for l in true_label:
                if l not in labels and len(labels)<args.episode_way:
                    labels.append(l)

            triplet_q = query.reshape(args.episode_shot, args.episode_way)
            triplet_qt = query_tmp.reshape(args.episode_shot, args.episode_way)
            positive = []
            negative = []

            for i in range(args.episode_way):
                row_index = random.randint(0, len(triplet_q) - 1)
                row_index_2 = random.randint(args.episode_way, len(triplet_q) - 1)
                row = matrix1[row_index]
                n_index = random.randint(0, args.episode_way)
                
                while i == n_index:
                    n_index = random.randint(0, args.episode_way)
                positive.append(triplet_q[row_index][i])
                negative.append(triplet_q[row_index_2][n_index])

                print(positive, negative)


            # 5w1s [5,512] to [1,5,512]
            # 5w5s [25,512] to [5,5,512]
            proto = proto.view(args.episode_shot, args.episode_way, proto.shape[-1])
            proto_tmp = proto_tmp.view(args.low_shot, args.low_way, proto.shape[-1])
            # get class prototype, mean(0) 0=index of shot number 
            proto = proto.mean(0).unsqueeze(0).unsqueeze(0)
            proto_tmp = proto_tmp.mean(0).unsqueeze(0).unsqueeze(0)
            #p 1,1,e_way,512  #q 1,1,q_shot,512
            query = query.unsqueeze(0).unsqueeze(0)
            query_tmp = query_tmp.unsqueeze(0).unsqueeze(0)

            protos = model.module.fc.weight[:args.base_class, :].clone().detach().unsqueeze(0).unsqueeze(0)
            # for l in labels:
            #     protos[0][0][l] = proto[0][0][labels.index(l)]

            protos = torch.cat([protos, proto_tmp], dim=2)
            query = torch.cat([query, query_tmp], dim=2)

            anchor = proto.view(args.episode_shot, proto.shape[-1])
            query_trip = query.view(args.episode_query*2, args.episode_way, query.shape[-1])

            logits = model.module._forward(protos, query)