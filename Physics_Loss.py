import torch
import torch.nn as nn

class LaEnergiaNoAparece(nn.Module):
    def __init__(self, L:float=0.1,thickness:float=0.001,board_k:float=15,ir_emmisivity:float=0.8):
        super(LaEnergiaNoAparece, self).__init__()
        
        # Seleccionar dispositivo dinámicamente
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        nx = 13
        ny = 13

        self.n_nodes = nx*ny # número total de nodos
        
        interfaces = [0,nx-1,nx*nx-1,nx*nx-nx]

        self.Boltzmann_cte = 5.67E-8

        # cálculo de los GLs y GRs
        dx = L/(nx-1)
        dy = L/(ny-1)
        GLx = thickness*board_k*dy/dx
        GLy = thickness*board_k*dx/dy
        GR = 2*dx*dy*ir_emmisivity

        # Generación de la matriz de acoplamientos conductivos [K]. 
        K_cols = []
        K_rows = []
        K_data = []
        for j in range(ny):
            for i in range(nx):
                id = i + nx*j
                if id in interfaces:
                    K_rows.append(id)
                    K_cols.append(id)
                    K_data.append(1)
                else:
                    GLii = 0
                    if i+1 < nx:
                        K_rows.append(id)
                        K_cols.append(id+1)
                        K_data.append(-GLx)
                        GLii += GLx
                    if i-1 >= 0:
                        K_rows.append(id)
                        K_cols.append(id-1)
                        K_data.append(-GLx)
                        GLii += GLx
                    if j+1 < ny:
                        K_rows.append(id)
                        K_cols.append(id+nx)
                        K_data.append(-GLy)
                        GLii += GLy
                    if j-1 >= 0:
                        K_rows.append(id)
                        K_cols.append(id-nx)
                        K_data.append(-GLy)
                        GLii += GLy
                    K_rows.append(id)
                    K_cols.append(id)
                    K_data.append(GLii)
        indices = torch.LongTensor([K_rows, K_cols])
        values = torch.FloatTensor(K_data)
        shape = torch.Size([self.n_nodes, self.n_nodes])
        self.K = torch.sparse_coo_tensor(indices, values, shape, dtype=torch.float)
        self.K = self.K.to(self.device)

        E_data = []
        E_id = []
        for id in range(self.n_nodes):
            if id not in interfaces:
                E_id.append(id)
                E_data.append(GR)
        indices = torch.tensor([E_id, E_id], dtype=torch.int64) 
        values = torch.tensor(E_data, dtype=torch.float32)  
        size = torch.Size([self.n_nodes, self.n_nodes])  
        self.E = torch.sparse_coo_tensor(indices, values, size)
        self.E = self.E.to(self.device)


    def forward(self, outputs, heaters, interfaces, Tenv):
                
        #Generación del vector Q
        heaters = torch.flatten(heaters, start_dim=1).to(self.device)

        interfaces = torch.flatten(interfaces, start_dim=1).to(self.device)

        Q = heaters + interfaces
        

        #Generación del vector T
        T = torch.flatten(outputs, start_dim=1)

        excessEnergy = torch.zeros((self.n_nodes,Q.size(0)), device=self.device)


        # T, Q, Tenv = T.cuda(), Q.cuda(), Tenv.cuda()
        # T = torch.transpose(T, 0, 1)
        # Q = torch.transpose(Q, 0, 1)
        Tenv = torch.transpose(Tenv, 0, 1).to(self.device)

        excessEnergy = torch.sparse.mm(self.K,T) + self.Boltzmann_cte*torch.sparse.mm(self.E,(T**4-Tenv**4))-Q

        return torch.mean(torch.abs(excessEnergy))
    
    
    
class PhysicsLossTransient(nn.Module):
    def __init__(self, 
                 L: float = 0.1,
                 thickness: float = 0.001,
                 board_k: float = 15,
                 ir_emmisivity: float = 0.8,
                 board_rho: float = 2700,
                 board_c: float = 900,
                 dt: float = 1
                 ):
        super(PhysicsLossTransient, self).__init__()
        
        # Seleccionar dispositivo dinámicamente
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        n = nx = ny = 13
        
        self.n_nodes = nx*ny # número total de nodos
        
        self.heater_node_map = {
            0: int((n-1)/4 + (n-1)/2 * n),
            1: int((n-1)/2 + (n-1)/4 * n),
            2: int((n-1)/4 + 3*(n-1)/4 * n),
            3: int(3*(n-1)/4 + 3*(n-1)/4 * n)
        }

        self.interface_node_map = {
            0: 0,
            1: n - 1,
            2: n * n - 1,
            3: n * n - n
        }

        interfaces = list(self.interface_node_map.values())

        self.Boltzmann_cte = 5.67E-8
        self.rho = board_rho
        self.cp = board_c
        self.dt = dt

        # cálculo de los GLs y GRs
        self.dx = L/(nx-1)
        self.dy = L/(ny-1)
        GLx = thickness*board_k*self.dy/self.dx
        GLy = thickness*board_k*self.dx/self.dy
        GR = 2*self.dx*self.dy*ir_emmisivity
        
        self.thickness = thickness

        # Generación de la matriz de acoplamientos conductivos [K]. 
        K_cols = []
        K_rows = []
        K_data = []
        for j in range(ny):
            for i in range(nx):
                id = i + nx*j
                if id in interfaces:
                    K_rows.append(id)
                    K_cols.append(id)
                    K_data.append(1)
                else:
                    GLii = 0
                    if i+1 < nx:
                        K_rows.append(id)
                        K_cols.append(id+1)
                        K_data.append(-GLx)
                        GLii += GLx
                    if i-1 >= 0:
                        K_rows.append(id)
                        K_cols.append(id-1)
                        K_data.append(-GLx)
                        GLii += GLx
                    if j+1 < ny:
                        K_rows.append(id)
                        K_cols.append(id+nx)
                        K_data.append(-GLy)
                        GLii += GLy
                    if j-1 >= 0:
                        K_rows.append(id)
                        K_cols.append(id-nx)
                        K_data.append(-GLy)
                        GLii += GLy
                    K_rows.append(id)
                    K_cols.append(id)
                    K_data.append(GLii)
                    
        indices = torch.LongTensor([K_rows, K_cols])
        values = torch.FloatTensor(K_data)
        shape = torch.Size([self.n_nodes, self.n_nodes])
        self.K = torch.sparse_coo_tensor(indices, values, shape, dtype=torch.float).to(self.device)

        E_data = []
        E_id = []
        
        for id in range(self.n_nodes):
            if id not in interfaces:
                E_id.append(id)
                E_data.append(GR)
                
        indices = torch.tensor([E_id, E_id], dtype=torch.int64) 
        values = torch.tensor(E_data, dtype=torch.float32)  
        size = torch.Size([self.n_nodes, self.n_nodes])  
        self.E = torch.sparse_coo_tensor(indices, values, size).to(self.device)
        
        
    def build_Q(self, heaters_t, interfaces_t):
        B = heaters_t.shape[0]
        Q = torch.zeros((B, self.n_nodes), device=self.device)

        for i in range(4):
            node_h = self.heater_node_map[i]
            node_i = self.interface_node_map[i]
            Q[:, node_h] += heaters_t[:, i]
            Q[:, node_i] += interfaces_t[:, i]

        return Q.transpose(0, 1)  # [n_nodes, B]


    def compute_step_loss(self, 
                T_new,     # Temperatura en el instante n+1 (predicha por la red, por ej)
                T_old,     # Temperatura en el instante n (condición conocida), 
                heaters_input, 
                interfaces_input, 
                Tenv):
        """
        T_new: tensor [batch_size, n_nodes] con la T en el paso n+1
        T_old: tensor [batch_size, n_nodes] con la T del paso n
        heaters, interfaces, Tenv: análogos a lo anterior.
        
        Devuelve la norma del 'residuo transitorio' según:
            [rho*c*(espesor)*(dx*dy)*(T_new - T_old)/dt  -  [ Q - K*T_old - sigma*E*(T_old^4 - Tenv^4) ] ]^2
        """

        #Generación del vector T
        T_new = T_new.flatten(start_dim=1).transpose(0,1).to(self.device)  # shape => [n_nodes, batch_size]
        T_old = T_old.flatten(start_dim=1).transpose(0,1).to(self.device)
        Tenv  = Tenv.flatten(start_dim=1).transpose(0,1).to(self.device)
        
        Q = self.build_Q(heaters_input, interfaces_input)  # [n_nodes, B]
        
        # Calculamos lado derecho del balance (igual que en el solver)
        # [Q - K*T_old - sigma*E*(T_old^4 - Tenv^4)]
        rhs =   Q \
                - torch.sparse.mm(self.K, T_old) \
                - self.Boltzmann_cte * torch.sparse.mm(self.E, (torch.pow(T_old, 4) - torch.pow(Tenv, 4)))
                  
        # Calculamos lado izquierdo transitorio:
        #  rho*c*(espesor)*(dx*dy)*(T_new - T_old)/dt
        vol_heat = self.rho * self.cp * self.thickness * (self.dx * self.dy)
        lhs_trans = vol_heat * (T_new - T_old) / self.dt

        # ExcessEnergy = LHS - RHS
        residual = lhs_trans - rhs  # [n_nodes, batch_size]

        return torch.mean(torch.abs(residual))
    
    def forward(self, T_pred, T_true, heaters_input, interfaces_input, Tenv):
        """
        T_pred:    [B, T, 1, 13, 13] → predicciones de temperatura
        T_true:    [B, T, 1, 13, 13] → ground truth (quizá usado para comparar)
        heaters_input: [B, 4]
        interfaces_input: [B, 4]
        Tenv:      [B, 1]
        """
        B, T, _, H, W = T_pred.shape
        
        if T < 2:
            raise ValueError("PhysicsLossTransient requires at least 2 time steps (T >= 2)")
    
        loss_total = 0.0

        for t in range(T - 1):  # hasta T-1 para comparar t vs t+1
            T_old = T_true[:, t, 0]     # [B, 13, 13]  
            T_new = T_pred[:, t+1, 0]   # [B, 13, 13]
            # heaters_t = heaters_input[:, t]
            # interfaces_t = interfaces_input[:, t]

            # Ahora llama a tu lógica física con T_old, T_new, etc.
            loss_t = self.compute_step_loss(T_new, T_old, heaters_input, interfaces_input, Tenv)

            loss_total += loss_t**2

        return loss_total / (T - 1)  # Promedio sobre todos los pasos
    
class BoundaryLoss(nn.Module):
    """
    Clase para calcular la pérdida asociada al no cumplimiento de las condiciones de contorno.
    """
    def __init__(self, nx=13, ny=13, device=None):
        super(BoundaryLoss, self).__init__()
        
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.n_nodes = nx * ny

        # Índices por defecto de los nodos de las 4 interfaces en una malla nx x ny
        self.interfaces = [0, nx - 1, nx * ny - 1, nx * ny - nx]

    def forward(self, T_pred, interfaces_target):
        """
        Calcula la pérdida como el error cuadrático medio en los nodos de interfaz.

        Args:
            T_pred (tensor): [batch_size, T, 1, nx, ny]
            interfaces_target (tensor): [batch_size, 4]

        Returns:
            Tensor escalar: pérdida media en los nodos de interfaz.
        """
        batch_size, T, _, nx, ny = T_pred.shape
        
        # Reorganizar T_pred: [batch_size, T, 1, nx, ny] -> [batch_size, T, 1, n_nodes]
        T_pred = T_pred.view(batch_size, T, 1, self.n_nodes)

        # Extraer solo los nodos de las interfaces: [batch_size, T, 1, 4] -> [batch_size, T, 4]
        T_interfaces = T_pred[:, :, 0, self.interfaces]

        # Asegurar que interfaces_target tiene el shape correcto
        interfaces_target = interfaces_target.unsqueeze(1).expand(-1, T, -1).to(self.device)  # [batch_size, T, 4]

        # Calcular MSE
        loss = torch.mean(torch.abs(T_interfaces - interfaces_target)**2)
        
        return loss
    
class TotalLoss(nn.Module):
    def __init__(self, 
                 mse_weight=1.0,
                 physics_weight=1.0,
                 boundary_weight=1.0,
                 denormalize_output_fn=None,
                 physics_params=None,
                 boundary_params=None):
        super(TotalLoss, self).__init__()

        # Pérdida de reconstrucción supervisada
        self.reconstruction = nn.MSELoss()
        self.lambda_rec = mse_weight
        self.lambda_phys = physics_weight
        self.lambda_bdry = boundary_weight
        self.denormalize = denormalize_output_fn

        # Parámetros por defecto si no se pasan
        if physics_params is None:
            physics_params = {}
        if boundary_params is None:
            boundary_params = {}

        # Construir funciones físicas y de contorno internamente
        self.physics_loss = PhysicsLossTransient(**physics_params)
        self.boundary_loss = BoundaryLoss(**boundary_params)

    def forward(self, y_hat, y, heaters, interfaces, Tenv):
        # MSE supervisado en espacio normalizado
        loss_rec = self.reconstruction(y_hat, y)

        # Denormalización antes de aplicar la física
        y_hat_denorm = self.denormalize(y_hat)
        y_denorm     = self.denormalize(y)

        # Física transitoria
        loss_phys = self.physics_loss(y_hat_denorm, y_denorm, heaters, interfaces, Tenv)

        # Condiciones de contorno
        loss_bdry = self.boundary_loss(y_hat_denorm, interfaces)

        # Combinación total
        total = self.lambda_rec * loss_rec + self.lambda_phys * loss_phys + self.lambda_bdry * loss_bdry
        return total, loss_rec, loss_phys, loss_bdry