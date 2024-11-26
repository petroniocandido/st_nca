DEFAULT_PATH = "."
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def checkpoint(modelo, arquivo):
  torch.save(modelo.state_dict(), DEFAULT_PATH + arquivo)

def checkpoint_all(modelo, otimizador, arquivo):
  torch.save({
    'optim': otimizador.state_dict(),
    'model': modelo.state_dict(),
}, DEFAULT_PATH + arquivo)

def resume(modelo, arquivo):
  modelo.load_state_dict(torch.load(DEFAULT_PATH + arquivo, weights_only=True, map_location=torch.device(DEVICE)))

def resume_all(modelo, otimizador, arquivo):
  checkpoint = torch.load(DEFAULT_PATH + arquivo, map_location=torch.device(DEVICE))
  modelo.load_state_dict(checkpoint['model'])
  otimizador.load_state_dict(checkpoint['optim'])