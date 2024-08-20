import torch
import torch.nn as nn
import math

class Encoder(nn.Module):
    def __init__(self, embedding_dim, h_dim, num_layers = 1, dropout=0.0):
        super(Encoder,self).__init__()
        
        self.embedding_dim = embedding_dim
        self.h_dim = h_dim
        self.num_layers = num_layers
        self.dropout = dropout
        
        self.encoder = nn.LSTM(input_size=self.embedding_dim, 
                               hidden_size=self.h_dim, 
                               num_layers=self.num_layers, 
                               dropout=self.dropout)
        
        self.spatial_embedding = nn.Linear(2, self.embedding_dim)
        
    def init_hidden(self,batch):
        return(
            torch.zeros(self.num_layers, batch, self.h_dim).cuda(),
            torch.zeros(self.num_layers, batch, self.h_dim).cuda()
        )
        
    def forward(self, obs_traj):
        """
        input:
        obs_traj -> tensor (obs_len,batch,2)
        
        output:
        final_h_state -> tensor (self.num_layers, batch, self.h_dim)
    
        """
        batch = obs_traj.size(1)
        
        #total barisnya sepanjang batch * obs_len
        #kolomnya sepanjang 2
        obs_traj_embedding = self.spatial_embedding(obs_traj.reshape(-1,2))
        
        # view itu ngerubah shape nya jadi (seq_len, batch, embedding dim)
        obs_traj_embedding = obs_traj_embedding.view(-1, batch, self.embedding_dim)
        
        state_tuple = self.init_hidden(batch)
        output, state = self.encoder(obs_traj_embedding, state_tuple)
        final_h_state = state[0]
        return final_h_state
    
class Decoder(nn.Module):
    def __init__(
        self, seq_pred_len, embedding_dim, h_dim, mlp_dim, 
        pool_every_time_step,bottleneck_dim, activation, 
        batch_norm, degree_of_vision,
        weight_pooling_features, num_layers = 1, dropout = 0.0
    ):
        super(Decoder,self).__init__()
        
        self.seq_pred_len = seq_pred_len
        self.h_dim = h_dim
        self.mlp_dim = mlp_dim
        self.pool_every_time_step = pool_every_time_step
        self.bottleneck_dim = bottleneck_dim
        self.activation = activation
        self.batch_norm = batch_norm
        self.embedding_dim = embedding_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.degree_of_vision = degree_of_vision
        self.weight_pooling_features = weight_pooling_features
        
        self.decoder = nn.LSTM(input_size=self.embedding_dim, 
                               hidden_size=self.h_dim, 
                               num_layers=self.num_layers, 
                               dropout=self.dropout)
        
        self.spatial_embedding = nn.Linear(2,self.embedding_dim)
        self.hidden2_pos_velocity = nn.Linear(self.h_dim, 4)
        
        if self.pool_every_time_step:
            self.pool_net = Pooling(
                embedding_dim=self.embedding_dim,
                h_dim=self.h_dim,
                mlp_dim=self.mlp_dim,
                bottleneck_dim=self.bottleneck_dim,
                activation=self.activation,
                batch_norm=self.batch_norm,
                degree_of_vision=self.degree_of_vision,
                weight_pooling_features=self.weight_pooling_features,
                dropout=self.dropout
            )
            
            mlp_dims = [self.h_dim + self.bottleneck_dim, self.mlp_dim, self.h_dim]
            self.mlp = self.make_mlp(
                dim_list=mlp_dims,
                activation=self.activation,
                batch_norm=self.batch_norm,
                dropout=self.dropout
            )
    def make_mlp(self,dim_list, activation, batch_norm, dropout):
        layers=[]
        for dim_in, dim_out in zip(dim_list[:-1], dim_list[1:]):
            layers.append(nn.Linear(dim_in,dim_out))
            if batch_norm:
                layers.append(nn.BatchNorm1d(dim_out))
                
            if activation == 'relu':
                layers.append(nn.ReLU())
            elif activation == 'leakyrelu':
                layers.append(nn.LeakyReLU())
            elif activation == 'tanh':
                layers.append(nn.Tanh())
            
            if(dropout > 0):
                layers.append(nn.Dropout(p=dropout))
                
        return nn.Sequential(*layers)
        
    def forward(self, last_pos, state_tuple, seq_start_end, last_pos_before):
        """
        inputs:
        last_pos -> tensor (batch,2)
        state_tuple -> hh,ch each has shape (self.num_layers, batch, self.h_dim)
        seq_start_end -> a list of tuples which delimit sequences within batch
        last_pos_before -> tensor (batch,2)
        
        output:
        pred_traj -> tensor (self.seq_len, batch, 2)
        """
        
        batch = last_pos.size(0)
        rel_last_post = last_pos - last_pos_before
        decoder_input = self.spatial_embedding(rel_last_post)
        decoder_input = decoder_input.view(1,batch,self.embedding_dim)
        
        pred_traj_fake = []
        for _ in range(self.seq_pred_len):
            output, state_tuple = self.decoder(decoder_input, state_tuple)
            curr_pos = self.hidden2_pos_velocity(output.view(-1,self.h_dim))
            curr_velocity = curr_pos[:,2:]
            rel_pos = curr_pos[:,:2]
            curr_pos = rel_pos+last_pos
            pred_traj_fake.append(curr_pos.view(batch,-1))
            decoder_input = self.spatial_embedding(rel_pos)
            decoder_input = decoder_input.view(1,batch,self.embedding_dim)
            before_pos = last_pos
            last_pos = curr_pos
            
            if self.pool_every_time_step:
                decoder_h = state_tuple[0]
                pool_h = self.pool_net(
                    decoder_h,
                    seq_start_end,
                    curr_pos,
                    curr_velocity,
                    before_pos
                )
                
                decoder_h = torch.cat([decoder_h.view(-1, self.h_dim), pool_h], dim=1)
                decoder_h = self.mlp(decoder_h)
                decoder_h = torch.unsqueeze(decoder_h, 0)
                state_tuple = (decoder_h, state_tuple[1])
            
        pred_traj_fake = torch.stack(pred_traj_fake, dim=0)
        return pred_traj_fake, state_tuple[0]
    
class Pooling(nn.Module):
    def __init__(
        self, embedding_dim, h_dim, mlp_dim, bottleneck_dim, 
        activation, batch_norm, degree_of_vision, weight_pooling_features, model,
        dropout=0.0,A=1, B1=2, B2=1
    ):
        super(Pooling,self).__init__()
        
        self.mlp_dim = mlp_dim
        self.h_dim = h_dim
        self.embedding_dim = embedding_dim
        self.activation = activation
        self.bottleneck_dim = bottleneck_dim
        self.dropout = dropout
        self.batch_norm = batch_norm
        self.degree_of_vision = degree_of_vision
        self.A = A
        self.B1 = B1
        self.B2 = B2
        self.model = model

        # weight [position, velocity, hidden_state]
        self.weight_pooling_features = weight_pooling_features
        
        mlp_pre_dim = self.embedding_dim + self.embedding_dim + self.h_dim
        mlp_pre_pool_dims = [mlp_pre_dim,512,self.bottleneck_dim]
        
        self.spatial_embedding = nn.Linear(2,self.embedding_dim)
        self.velocity_embedding = nn.Linear(2,self.embedding_dim)

        self.mlp_pre_pool = self.make_mlp(
            mlp_pre_pool_dims,
            activation=activation,
            batch_norm=batch_norm,
            dropout=dropout)
        
        mlp_pool_h_dim = [self.bottleneck_dim + self.bottleneck_dim, self.bottleneck_dim]
        self.mlp_pool_h = self.make_mlp(
            mlp_pool_h_dim,
            activation=activation,
            batch_norm=batch_norm,
            dropout=dropout
        )
        
    def check_inside_egg(self, x,y,a,b1,b2):
        # assume point is not in the egg
        res = 2

        # calculate the equation
        if(y >= 0):
            res = ((x*x)/(a*a))+((y*y)/(b1*b1))
        else:
            res = ((x*x)/(a*a))+((y*y)/(b2*b2))

        #check if the point is inside the egg
        if(res <= 1):
            return True
        return False

    def cross_product(self,x1,y1,x2,y2):
        # if negative then p1 ccw from p2
        # if positive then p1 cw from p2
        # if 0 then p1 is collinear from p2
        return (x1*y2)-(x2*y1)

    def check_cone(self,degree,x,y):
        if(degree < 180):

            halfDegree = degree/2.0

            # a is samping, b is depan, c is miring
            a = 2
            # using cos
            c = a / math.cos(math.radians(halfDegree))
            b = math.sin(math.radians(halfDegree)) * c

            # 0,x1 is first point of the end of the cone
            x1 = 0-b

            # 0,x2 is second point of the end of the cone
            x2 = 0+b

            # if turn right and turn left
            if((self.cross_product(x,y,x1,a) > 0) and (self.cross_product(x,y,x2,a) < 0)):
                return True

            # if collinear
            if((self.cross_product(x,y,x1,2) == 0) or (self.cross_product(x,y,x2,2) == 0)):
                return True

        elif(degree == 180):
            x1 = -2
            x2 = 2
            # if turn right and turn left
            if((self.cross_product(x,y,x1,0) > 0) and (self.cross_product(x,y,x2,0) < 0)):
                return True

            # if collinear
            if((self.cross_product(x,y,x1,0) == 0) or (self.cross_product(x,y,x2,0) == 0)):
                return True

        else :
            degree = degree - 180
            halfDegree = 90 - (degree/2.0)

            # a is samping, b is depan, c is miring
            a = 2
            c = a / math.cos(math.radians(halfDegree))
            b = math.sin(math.radians(halfDegree)) * c

            # 0,y1 is first point of the end of the cone
            x1 = 0-b

            # 0,y2 is second point of the end of the cone
            x2 = 0+b

            # if turn left and turn right
            if((self.cross_product(x,y,x1,-1*a) < 0) and (self.cross_product(x,y,x2,-1*a) > 0)):
                return False

            return True


        # not inside the cone
        return False

    def check_inside_egg_and_cone(self,degree,a,b1,b2,x,y):
        # if not inside egg definitely false
        if(not (self.check_inside_egg(x,y,a,b1,b2))):
            return False

        # if checking upper part of the egg
        if(degree <= 180):
            # if it is on the lower part of the egg, definitely false
            if(y < 0):
                return False


        # if inside the cone, definitely false
        if(not(self.check_cone(degree,x,y))):
            return False

        # other than all condition above, it is true
        return True

    def check_point_on_egg_and_cone(self, degree, x, y, A = 1, B1 = 2, B2 = 1):
        """
        The parameter is degree, a, b1, b2, x, y.
        a = 1, b1 = 2, b2 = 1
        """
        return self.check_inside_egg_and_cone(degree,A,B1,B2,x,y)
    
    def rotate_coordinate(self,x,y,degree):
        #degree in radians
        x_trans = (math.cos(degree)*x)+(math.sin(degree)*(-1)*y)
        y_trans = (math.sin(degree)*x)+(math.cos(degree)*y)
        return x_trans,y_trans

    def calculate_degree(self,x,y,x_before, y_before):
        x_rel = x - x_before
        y_rel = y - y_before
        if x_rel > 0 and y_rel > 0:
            return math.atan(abs(y_rel)/abs(x_rel)) + math.radians(270)
        elif x_rel < 0 and y_rel > 0:
            return math.radians(90) - math.atan(abs(y_rel)/abs(x_rel))
        elif x_rel < 0 and y_rel < 0:
            return math.radians(90) + math.atan(abs(y_rel)/abs(x_rel))
        elif x_rel > 0 and y_rel < 0:
            return math.radians(270) - math.atan(abs(y_rel)/abs(x_rel))
        elif x_rel == 0 and y_rel >0:
            return math.radians(0)
        elif x_rel == 0 and y_rel < 0:
            return math.radians(180)
        elif y_rel == 0 and x_rel >0:
            return math.radians(270)
        elif y_rel == 0 and x_rel < 0:
            return math.radians(90)
        return math.radians(0)
    
    def make_mlp(self,dim_list, activation, batch_norm, dropout):
        layers=[]
        for dim_in, dim_out in zip(dim_list[:-1], dim_list[1:]):
            layers.append(nn.Linear(dim_in,dim_out))
            if batch_norm:
                layers.append(nn.BatchNorm1d(dim_out))
                
            if activation == 'relu':
                layers.append(nn.ReLU())
            elif activation == 'leakyrelu':
                layers.append(nn.LeakyReLU())
            elif activation == 'tanh':
                layers.append(nn.Tanh())
            
            if(dropout > 0):
                layers.append(nn.Dropout(p=dropout))
                
        return nn.Sequential(*layers)

    def forward(self, h_states, seq_start_end, end_pos, end_velocity, before_end_pos):
        """
        Inputs:
        - h_states: Tensor of shape (num_layers, batch, h_dim)
        - seq_start_end: A list of tuples which delimit sequences within batch
        - end_pos: Tensor of shape (batch, 2)
        - end_velocity : Tensor of shape (batch, 2)
        - before_end_pos : Tensor of shape (batch, 2)
        Output:
        - pool_h: Tensor of shape (batch, bottleneck_dim)
        """
        ###############
        pool_h_max = []
        pool_h_min = []
        ###############
        for _, (start, end) in enumerate(seq_start_end):
            start = start.item()
            end = end.item()
            num_ped = end - start
            curr_hidden = h_states.view(-1, self.h_dim)[start:end]
            curr_end_pos = end_pos[start:end]
            curr_end_velocity = end_velocity[start:end]
            curr_before_end_pos = before_end_pos[start:end]  

            
            for i in range(num_ped):
                with torch.no_grad():
                    degree_direction = self.calculate_degree(
                        curr_end_pos[i][0],
                        curr_end_pos[i][1],
                        curr_before_end_pos[i][0],
                        curr_before_end_pos[i][1]
                    )
                    curr_hidden_considered = []
                    curr_end_pos_considered = []
                    curr_end_velocity_considered = []
                    counter = 0
                    for j in range(num_ped):
                        if j != i:
                            x_trans, y_trans = self.rotate_coordinate(curr_end_pos[j][0] - curr_end_pos[i][0], curr_end_pos[j][1] - curr_end_pos[i][1], degree_direction)
                            if self.check_point_on_egg_and_cone(self.degree_of_vision,x_trans,y_trans):
                                counter+=1
                                curr_hidden_considered.append(curr_hidden[j])
                                curr_end_pos_considered.append(curr_end_pos[j])
                                curr_end_velocity_considered.append(curr_end_velocity[j])
                                
                curr_end_pos_considered_rel = torch.zeros(2,2).cuda()
                curr_end_velocity_considered_rel = torch.zeros(2,2).cuda()

                if counter > 0:
                    curr_end_pos_considered = torch.stack(curr_end_pos_considered,dim=0)
                    curr_end_velocity_considered = torch.stack(curr_end_velocity_considered,dim=0)
                    curr_hidden_considered = torch.stack(curr_hidden_considered,dim=0)

                    # hitung posisi relatif dari tetangga
                    curr_end_pos_considered_rel = curr_end_pos_considered - curr_end_pos[i]

                    # hitung kecepatan relatif dari tetangga
                    curr_end_velocity_considered_rel = curr_end_velocity_considered - curr_end_velocity[i]

                    # pemberian weight pada fitur
                    curr_end_pos_considered_rel = curr_end_pos_considered_rel * self.weight_pooling_features[0]
                    curr_end_velocity_considered_rel = curr_end_velocity_considered_rel * self.weight_pooling_features[1]
                    curr_hidden_considered = curr_hidden_considered * self.weight_pooling_features[2]

                if self.model == 2:

                    curr_pos_rel_embedding = self.spatial_embedding(curr_end_pos_considered_rel)

                    curr_velocity_rel_embedding = self.velocity_embedding(curr_end_velocity_considered_rel)

                    if counter>0:
                        mlp_h_input = torch.cat([curr_pos_rel_embedding,curr_velocity_rel_embedding,curr_hidden_considered], dim=1)
                    else:
                        curr_hidden_considered = torch.zeros(2,h_states.size(2)).cuda()
                        mlp_h_input = torch.cat([curr_pos_rel_embedding,curr_velocity_rel_embedding,curr_hidden_considered],dim=1)

                    curr_pool_h = self.mlp_pre_pool(mlp_h_input)
            
                    pool_h_max.append(torch.max(curr_pool_h,dim=0).values)
                    pool_h_min.append(torch.min(curr_pool_h,dim=0).values)
                else:
                    if counter>0:
                        curr_pos_rel_embedding = self.spatial_embedding(curr_end_pos_considered_rel)

                        curr_velocity_rel_embedding = self.velocity_embedding(curr_end_velocity_considered_rel)

                        mlp_h_input = torch.cat([curr_pos_rel_embedding,curr_velocity_rel_embedding,curr_hidden_considered], dim=1)

                        curr_pool_h = self.mlp_pre_pool(mlp_h_input)
                        ############
                        pool_h_max.append(torch.max(curr_pool_h,dim=0).values)
                        pool_h_min.append(torch.min(curr_pool_h,dim=0).values)
                        ###########
                    else :
                        ##########
                        pool_h_max.append(torch.zeros(self.bottleneck_dim).cuda())
                        pool_h_min.append(torch.zeros(self.bottleneck_dim).cuda())
                  
                    
        pool_h_max = torch.stack(pool_h_max,dim=0)
        pool_h_min = torch.stack(pool_h_min,dim=0)
        mix_pool = torch.cat([pool_h_max,pool_h_min],dim=1)
        pool_h = self.mlp_pool_h(mix_pool)
        return pool_h
    
    
class TrajectoryGenerator(nn.Module):
    def __init__(
        self, obs_len, pred_len, embedding_dim, encoder_h_dim, noise_type,
        decoder_h_dim, mlp_dim, noise_dim, dropout, bottleneck_dim, pool_every_time_step,
        weight_pooling_features, degree_of_vision, activation, batch_norm, model,num_layers=1
    ):
        super(TrajectoryGenerator,self).__init__()
        self.obs_len = obs_len
        self.pred_len = pred_len
        self.embedding_dim = embedding_dim
        self.encoder_h_dim = encoder_h_dim
        self.decoder_h_dim = decoder_h_dim
        self.mlp_dim = mlp_dim
        self.noise_dim = noise_dim
        self.dropout = dropout
        self.bottleneck_dim = bottleneck_dim
        self.activation = activation
        self.batch_norm = batch_norm
        self.num_layers = num_layers
        self.weight_pooling_features = weight_pooling_features
        self.degree_of_vision = degree_of_vision
        self.pool_every_time_step = pool_every_time_step
        self.noise_type = noise_type
        self.model = model
        
        self.encoder = Encoder(
            embedding_dim=self.embedding_dim,
            h_dim=self.encoder_h_dim,
            num_layers=self.num_layers,
            dropout=self.dropout
        )
        
        self.decoder = Decoder(
            seq_pred_len=self.pred_len,
            embedding_dim=self.embedding_dim,
            h_dim=self.decoder_h_dim,
            mlp_dim=self.mlp_dim,
            pool_every_time_step=self.pool_every_time_step,
            bottleneck_dim=self.bottleneck_dim,
            activation=self.activation,
            batch_norm=self.batch_norm,
            degree_of_vision=self.degree_of_vision,
            weight_pooling_features=self.weight_pooling_features,
            num_layers=self.num_layers,
            dropout=self.dropout
        )
        
        self.poolnet = Pooling(
            embedding_dim=self.embedding_dim,
            h_dim=self.encoder_h_dim,
            mlp_dim=self.mlp_dim,
            bottleneck_dim=self.bottleneck_dim,
            activation=self.activation,
            batch_norm=self.batch_norm,
            degree_of_vision=self.degree_of_vision,
            weight_pooling_features=self.weight_pooling_features,
            dropout=self.dropout,
            model=model
        )
        
        if self.noise_dim[0] == 0:
            self.noise_dim = None
        else :
            self.noise_first_dim = noise_dim[0]
            
        if model == 1:
            input_dim = self.encoder_h_dim + self.bottleneck_dim
        
            mlp_decoder_context_dim = [input_dim,self.mlp_dim, self.decoder_h_dim-self.noise_first_dim]
            
            self.mlp_decoder_context = self.make_mlp(
                mlp_decoder_context_dim,
                activation=self.activation, 
                batch_norm=self.batch_norm,
                dropout=self.dropout
            )
            
    def make_mlp(self,dim_list, activation, batch_norm, dropout):
        layers=[]
        for dim_in, dim_out in zip(dim_list[:-1], dim_list[1:]):
            layers.append(nn.Linear(dim_in,dim_out))
            if batch_norm:
                layers.append(nn.BatchNorm1d(dim_out))
                
            if activation == 'relu':
                layers.append(nn.ReLU())
            elif activation == 'leakyrelu':
                layers.append(nn.LeakyReLU())
            elif activation == 'tanh':
                layers.append(nn.Tanh())
            
            if(dropout > 0):
                layers.append(nn.Dropout(p=dropout))
                
        return nn.Sequential(*layers)
            
    def get_noise(self,shape, noise_type):
        if noise_type == 'gaussian':
            return torch.randn(*shape).cuda()
        elif noise_type == 'uniform':
            return torch.rand(*shape).sub_(0.5).mul_(2.0).cuda()
        raise ValueError('Unrecognized noise type "%s"' % noise_type)
        
    def add_noise(self, _input, seq_start_end):
        """
        Inputs:
        - _input: Tensor
        - seq_start_end: A list of tuples which delimit sequences within batch
        Outputs:
        - decoder_h: Tensor of shape (_, decoder_h_dim)
        """
        if not self.noise_dim:
            return _input

        
        noise_shape = (seq_start_end.size(0), ) + self.noise_dim
        
        z_decoder = self.get_noise(noise_shape, self.noise_type)

        _list = []
        for idx, (start, end) in enumerate(seq_start_end):
            start = start.item()
            end = end.item()
            _vec = z_decoder[idx].view(1, -1)
            _to_cat = _vec.repeat(end - start, 1)
            _list.append(torch.cat([_input[start:end], _to_cat], dim=1))
        decoder_h = torch.cat(_list, dim=0)
        return decoder_h
    
    def forward(self, obs_traj, seq_start_end):
        """
        Inputs:
        - obs_traj: Tensor of shape (obs_len, batch, 4)
        - seq_start_end: A list of tuples which delimit sequences within batch.
        Output:
        - pred_traj_fake: Tensor of shape (self.pred_len, batch, 2)
        """
        batch = obs_traj.size(1)
        # Encode seq
        temp_obs_traj = obs_traj[:, :, :2]
        temp_obs_traj_rel = torch.zeros_like(temp_obs_traj)
        with torch.no_grad():
            temp_obs_traj_rel [1:,:,:] =  temp_obs_traj[1:,:,:]-temp_obs_traj[:-1,:,:]
        final_encoder_h = self.encoder(temp_obs_traj_rel)
        
        # Pool States
        last_obs = obs_traj[-1]
        last_pos = last_obs[:,:2]
        last_velocity = last_obs[:,2:]
        last_obs_before = obs_traj[-2]
        last_pos_before = last_obs_before[:,:2]

        pool_h = self.poolnet(final_encoder_h, seq_start_end, last_pos, last_velocity, last_pos_before)
        
        # Construct input hidden states for decoder
        # concat final_encoder_h dan pool_h
        mlp_decoder_context_input = torch.cat(
            [final_encoder_h.view(-1, self.encoder_h_dim), pool_h], dim=1)

        # Add Noise
        if self.model == 1:
            noise_input = self.mlp_decoder_context(mlp_decoder_context_input)
        else:
            noise_input = mlp_decoder_context_input

        decoder_h = self.add_noise(noise_input,seq_start_end=seq_start_end)
        
        decoder_h = torch.unsqueeze(decoder_h, 0)

        decoder_c = torch.zeros(
            self.num_layers, batch, self.decoder_h_dim
        ).cuda()

        state_tuple = (decoder_h, decoder_c)
        
        # Predict Trajectory
        decoder_out = self.decoder(
            last_pos,
            state_tuple,
            seq_start_end,
            last_pos_before
        )
        pred_traj_fake, final_decoder_h = decoder_out

        return pred_traj_fake
    
class TrajectoryDiscriminator(nn.Module):
    def __init__(
            self, obs_len, pred_len, embedding_dim, 
            h_dim, mlp_dim, num_layers, activation, 
            batch_norm, dropout
        ):
        super(TrajectoryDiscriminator,self).__init__()
        
        self.obs_len = obs_len
        self.pred_len = pred_len
        self.embedding_dim = embedding_dim
        self.h_dim = h_dim
        self.mlp_dim = mlp_dim
        self.num_layers = num_layers
        self.activation = activation
        self.batch_norm = batch_norm
        self.dropout = dropout
        
        self.encoder = Encoder(
            embedding_dim=self.embedding_dim,
            h_dim=self.h_dim,
            num_layers=self.num_layers,
            dropout=self.dropout
        )
        
        real_classifier_dims = [self.h_dim, self.mlp_dim, 1]
        
        self.real_classifier = self.make_mlp(
            real_classifier_dims, 
            activation='relu', 
            batch_norm=self.batch_norm, 
            dropout=self.dropout
        )
        
    def make_mlp(self,dim_list, activation, batch_norm, dropout):
        layers=[]
        for dim_in, dim_out in zip(dim_list[:-1], dim_list[1:]):
            layers.append(nn.Linear(dim_in,dim_out))
            if batch_norm:
                layers.append(nn.BatchNorm1d(dim_out))
                
            if activation == 'relu':
                layers.append(nn.ReLU())
            elif activation == 'leakyrelu':
                layers.append(nn.LeakyReLU())
            elif activation == 'tanh':
                layers.append(nn.Tanh())
            
            if(dropout > 0):
                layers.append(nn.Dropout(p=dropout))
                
        return nn.Sequential(*layers)
        
    def forward(self, traj):
        """
        Inputs:
        - traj: Tensor of shape (obs_len + pred_len, batch, 2)
        Output:
        - scores: Tensor of shape (batch,) with real/fake scores
        """
        final_h = self.encoder(traj)
       
        classifier_input = final_h.squeeze()
       
        scores = self.real_classifier(classifier_input)
        return scores