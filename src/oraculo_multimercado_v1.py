import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model
import xgboost as xgb
from scipy.stats import poisson
import itertools
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

print("========================================================")
print("🌍 ORÁCULO TOTAL: PRECISIÓN MÁXIMA + CONTABILIDAD EXCEL 🌍")
print("========================================================")

# ==========================================
# RUTAS DINÁMICAS (Para la estructura de GitHub)
# ==========================================
# Detecta automáticamente la raíz del proyecto
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, 'models', 'v1_ensamble_multimercado')
DATA_DIR = os.path.join(BASE_DIR, 'data', 'processed')

# Configuración de inversor
MI_BANKROLL_TOTAL = 1000.0  
KELLY_FRACCIONAL = 0.25     
UMBRAL_EV_RESULTADOS = 0.0801 
UMBRAL_EV_GOLES = 0.0986      
UMBRAL_PARLAY = 0.05          
MAX_SELECCIONES = 3         

# ==========================================
# CARGAR LOS MODELOS (Usando rutas absolutas)
# ==========================================
print("🧠 Cargando Inteligencias Artificiales...")
try:
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
    scaler_x = joblib.load(os.path.join(MODELS_DIR, 'oraculo_xG.pkl'))
    modelo_xgb_x = xgb.XGBClassifier()
    modelo_xgb_x.load_model(os.path.join(MODELS_DIR, 'oraculoxG_xgb.json'))
    modelo_nn_x = load_model(os.path.join(MODELS_DIR, 'oraculoxG_nn.keras'))
    
    scaler_g = joblib.load(os.path.join(MODELS_DIR, 'oraculo_goles_scaler.pkl'))
    modelo_xgb_g = xgb.XGBClassifier()
    modelo_xgb_g.load_model(os.path.join(MODELS_DIR, 'oraculo_goles_xgb.json'))
    modelo_nn_g = load_model(os.path.join(MODELS_DIR, 'oraculo_goles_nn.keras'))
    print("   ✅ IA de Resultados e IA de Goles en línea.")
except Exception as e:
    print(f"   ❌ ERROR FATAL al cargar modelos: {e}")

# ==========================================
# CARGAR BASE DE DATOS PROCESADA
# ==========================================
print("📊 Cargando base de datos histórica (df_super)...")
try:
    df_super = pd.read_csv(os.path.join(DATA_DIR, 'df_super.csv'))
    df_super['Date'] = pd.to_datetime(df_super['Date'])
    print("   ✅ Base de datos lista.")
except Exception as e:
    print(f"   ❌ ERROR al cargar datos: {e}")
    print("   ⚠️ Asegúrate de haber exportado 'df_super.csv' a la carpeta 'data/processed/'")
    df_super = pd.DataFrame()

# Partidos de la jornada 
partidos_jornada = [
    {'Local': 'Celta', 'Visita': 'Alaves', 'Odd_1': 1.93, 'Odd_X': 3.27, 'Odd_2': 4.23, 'Odd_Over': 2.10, 'Odd_Under': 1.70},
    {'Local': 'Ath Bilbao', 'Visita': 'Betis', 'Odd_1': 2.09, 'Odd_X': 3.43, 'Odd_2': 3.44, 'Odd_Over': 1.95, 'Odd_Under': 1.80},
    {'Local': 'Real Madrid', 'Visita': 'Ath Madrid', 'Odd_1': 1.82, 'Odd_X': 4.00, 'Odd_2': 3.86, 'Odd_Over': 1.55, 'Odd_Under': 2.25},
    {'Local': 'Roma', 'Visita': 'Lecce', 'Odd_1': 1.46, 'Odd_X': 4.31, 'Odd_2': 6.93, 'Odd_Over': 1.65, 'Odd_Under': 2.20},
    {'Local': 'Fiorentina', 'Visita': 'Inter', 'Odd_1': 5.05, 'Odd_X': 4.12, 'Odd_2': 1.62, 'Odd_Over': 1.70, 'Odd_Under': 2.10},
    {'Local': 'Mainz', 'Visita': 'Ein Frankfurt', 'Odd_1': 2.15, 'Odd_X': 3.52, 'Odd_2': 3.22, 'Odd_Over': 1.78, 'Odd_Under': 2.00},
    {'Local': 'St Pauli', 'Visita': 'Freiburg', 'Odd_1': 2.61, 'Odd_X': 3.04, 'Odd_2': 2.87, 'Odd_Over': 2.30, 'Odd_Under': 1.60},
    {'Local': 'Augsburg', 'Visita': 'Stuttgart', 'Odd_1': 3.28, 'Odd_X': 3.77, 'Odd_2': 2.04, 'Odd_Over': 1.57, 'Odd_Under': 2.30},
    {'Local': 'Marseille', 'Visita': 'Lille', 'Odd_1': 1.83, 'Odd_X': 3.72, 'Odd_2': 4.11, 'Odd_Over': 1.78, 'Odd_Under': 2.00},
    {'Local': 'Paris FC', 'Visita': 'Le Havre', 'Odd_1': 1.98, 'Odd_X': 3.27, 'Odd_2': 4.01, 'Odd_Over': 2.05, 'Odd_Under': 1.75},
    {'Local': 'Rennes', 'Visita': 'Metz', 'Odd_1': 1.30, 'Odd_X': 5.68, 'Odd_2': 8.77, 'Odd_Over': 1.54, 'Odd_Under': 2.50},
    {'Local': 'Nantes', 'Visita': 'Strasbourg', 'Odd_1': 3.65, 'Odd_X': 3.57, 'Odd_2': 1.98, 'Odd_Over': 1.95, 'Odd_Under': 1.85}
]

def oraculo_maestro_contable(lista_partidos, df_historico, bankroll):
    print("\n" + "="*56)
    print("🔍 ESCANEANDO MULTI-MERCADOS Y CONSTRUYENDO PORTAFOLIO")
    print("="*56)
    
    inversion_total = 0.0
    candidatos_parlay = [] 
    
    log_apuestas_simples = []
    log_apuestas_parlays = []
    fecha_ejecucion = datetime.now().strftime("%Y-%m-%d %H:%M")
    
    for p in lista_partidos:
        local, visita = p['Local'], p['Visita']
        odd_L, odd_E, odd_V = p['Odd_1'], p['Odd_X'], p['Odd_2']
        odd_Ov, odd_Un = p['Odd_Over'], p['Odd_Under']
        
        try:
            # extracción de datos 
            df_L = df_historico[(df_historico['HomeTeam'] == local) | (df_historico['AwayTeam'] == local)].sort_values('Date')
            df_V = df_historico[(df_historico['HomeTeam'] == visita) | (df_historico['AwayTeam'] == visita)].sort_values('Date')
            
            if df_L.empty or df_V.empty: 
                print(f"   ⚠️ OMITIDO: {local} vs {visita} (Nombres no encontrados)")
                continue
                
            forma_L = df_L.iloc[-1]
            forma_V = df_V.iloc[-1]
            elo_diff = (forma_L['Elo_Home'] + 50) - forma_V['Elo_Away']
            
            proj_xg_home = (forma_L['Home_xG_For_Form'] + forma_V['Away_xG_Ag_Form']) / 2.0
            proj_xg_away = (forma_V['Away_xG_For_Form'] + forma_L['Home_xG_Ag_Form']) / 2.0
            
            p_H = [poisson.pmf(k, proj_xg_home) for k in range(6)]
            p_A = [poisson.pmf(k, proj_xg_away) for k in range(6)]
            matriz = np.outer(p_H, p_A)
            
            # vectores de entrada
            input_1x2 = {
                'Elo_Diff': elo_diff, 'Elo_Home': forma_L['Elo_Home'], 'Elo_Away': forma_V['Elo_Away'],
                'Odds_Home': odd_L, 'Odds_Draw': odd_E, 'Odds_Away': odd_V,
                'Home_Shots_Total_For_Form': forma_L['Home_Shots_Total_For_Form'], 'Home_Shots_Target_For_Form': forma_L['Home_Shots_Target_For_Form'],
                'Home_Goals_For_Form': forma_L['Home_Goals_For_Form'], 'Home_Goals_Ag_Form': forma_L['Home_Goals_Ag_Form'], 'Home_Pts_Form': forma_L['Home_Pts_Form'],
                'Away_Shots_Total_For_Form': forma_V['Away_Shots_Total_For_Form'], 'Away_Shots_Target_For_Form': forma_V['Away_Shots_Target_For_Form'],
                'Away_Goals_For_Form': forma_V['Away_Goals_For_Form'], 'Away_Goals_Ag_Form': forma_V['Away_Goals_Ag_Form'], 'Away_Pts_Form': forma_V['Away_Pts_Form'],
                'Proj_xG_Home': proj_xg_home, 'Proj_xG_Away': proj_xg_away,
                'Poisson_Prob_1': np.tril(matriz, -1).sum(), 'Poisson_Prob_X': np.trace(matriz), 
                'Poisson_Prob_2': np.triu(matriz, 1).sum(), 
                'Poisson_Over25': 1 - (matriz[0,0] + matriz[1,0] + matriz[0,1] + matriz[1,1] + matriz[2,0] + matriz[0,2])
            }
            X_1x2 = pd.DataFrame([input_1x2])
            
            input_goles = {
                'Proj_xG_Home': proj_xg_home, 'Proj_xG_Away': proj_xg_away,
                'Poisson_Over25': input_1x2['Poisson_Over25'], 'Poisson_Prob_X': input_1x2['Poisson_Prob_X'],
                'Home_Goals_For_Form': forma_L['Home_Goals_For_Form'], 'Home_Goals_Ag_Form': forma_L['Home_Goals_Ag_Form'],
                'Away_Goals_For_Form': forma_V['Away_Goals_For_Form'], 'Away_Goals_Ag_Form': forma_V['Away_Goals_Ag_Form'],
                'Home_Shots_Target_For_Form': forma_L['Home_Shots_Target_For_Form'], 'Away_Shots_Target_For_Form': forma_V['Away_Shots_Target_For_Form']
            }
            X_goles = pd.DataFrame([input_goles])
            
            # predicciones 
            p_L, p_E, p_V = (modelo_xgb_x.predict_proba(X_1x2)[0] + modelo_nn_x.predict(scaler_x.transform(X_1x2), verbose=0)[0]) / 2.0
            p_Un, p_Ov = (modelo_xgb_g.predict_proba(X_goles)[0] + modelo_nn_g.predict(scaler_g.transform(X_goles), verbose=0)[0]) / 2.0
            
            odd_1X = (odd_L * odd_E) / (odd_L + odd_E)
            odd_X2 = (odd_E * odd_V) / (odd_E + odd_V)
            
            ev_L, ev_V = (p_L * odd_L) - 1, (p_V * odd_V) - 1
            ev_1X, ev_X2 = ((p_L + p_E) * odd_1X) - 1, ((p_E + p_V) * odd_X2) - 1
            ev_Ov, ev_Un = (p_Ov * odd_Ov) - 1, (p_Un * odd_Un) - 1

            print(f"\n🏟️ {local} vs {visita}")
            
            # análisis de apuestas simples con su respectivo umbral
            def procesar_simple(prob, odd, ev, mercado, umbral_especifico, max_pct=0.05):
                nonlocal inversion_total
                if ev > umbral_especifico:
                    stake = min(max(0, (ev / (odd - 1)) * KELLY_FRACCIONAL), max_pct)
                    monto = bankroll * stake
                    if monto > 0:
                        inversion_total += monto
                        print(f"   🟢 SIMPLE: {mercado} (@{odd:.2f}) | EV: +{ev*100:.1f}% | INVERTIR: €{monto:.2f}")
                        
                        log_apuestas_simples.append({
                            'Fecha_Analisis': fecha_ejecucion,
                            'Partido': f"{local} vs {visita}",
                            'Mercado': mercado,
                            'Cuota': round(odd, 2),
                            'Prob_IA': f"{prob*100:.2f}%",
                            'EV': f"+{ev*100:.2f}%",
                            'Inversion_Euros': round(monto, 2),
                            'Resultado_Real': '' 
                        })

            # Aplicando los umbrales específicos:
            if p_L > 0.45: procesar_simple(p_L, odd_L, ev_L, "GANA LOCAL", UMBRAL_EV_RESULTADOS)
            procesar_simple(p_L + p_E, odd_1X, ev_1X, "DOBLE 1X", UMBRAL_EV_RESULTADOS)
            if p_V > 0.45: procesar_simple(p_V, odd_V, ev_V, "GANA VISITA", UMBRAL_EV_RESULTADOS)
            procesar_simple(p_E + p_V, odd_X2, ev_X2, "DOBLE X2", UMBRAL_EV_RESULTADOS)
            
            # Goles con su barrera del 9.86%
            procesar_simple(p_Ov, odd_Ov, ev_Ov, "OVER 2.5", UMBRAL_EV_GOLES, max_pct=0.04) 
            procesar_simple(p_Un, odd_Un, ev_Un, "UNDER 2.5", UMBRAL_EV_GOLES, max_pct=0.04)

            # Colección de selecciones probables para combinadas 
            if p_L + p_E > 0.60: candidatos_parlay.append({'partido': f"{local} vs {visita}", 'mercado': f"{local} (1X)", 'prob': p_L + p_E, 'odd': odd_1X})
            if p_E + p_V > 0.60: candidatos_parlay.append({'partido': f"{local} vs {visita}", 'mercado': f"{visita} (X2)", 'prob': p_E + p_V, 'odd': odd_X2})
            if p_L > 0.55: candidatos_parlay.append({'partido': f"{local} vs {visita}", 'mercado': f"Gana {local}", 'prob': p_L, 'odd': odd_L})
            if p_V > 0.55: candidatos_parlay.append({'partido': f"{local} vs {visita}", 'mercado': f"Gana {visita}", 'prob': p_V, 'odd': odd_V})
            
            if p_Ov > 0.58: candidatos_parlay.append({'partido': f"{local} vs {visita}", 'mercado': "Over 2.5 Goles", 'prob': p_Ov, 'odd': odd_Ov})
            if p_Un > 0.58: candidatos_parlay.append({'partido': f"{local} vs {visita}", 'mercado': "Under 2.5 Goles", 'prob': p_Un, 'odd': odd_Un})

        except Exception as e:
            continue
            
    # Juntar opciones probables para parlays 
    parlays_encontrados = []
    
    for k in range(2, MAX_SELECCIONES + 1):
        if len(candidatos_parlay) < k: break
            
        for combo in itertools.combinations(candidatos_parlay, k):
            partidos_en_ticket = [apuesta['partido'] for apuesta in combo]
            if len(set(partidos_en_ticket)) < len(partidos_en_ticket): continue 
                
            prob_comb = np.prod([apuesta['prob'] for apuesta in combo])
            odd_comb = np.prod([apuesta['odd'] for apuesta in combo])
            ev_comb = (prob_comb * odd_comb) - 1
            
            if ev_comb > UMBRAL_PARLAY:
                parlays_encontrados.append({'legs': k, 'combo': combo, 'prob': prob_comb, 'odd': odd_comb, 'ev': ev_comb})

    if parlays_encontrados:
        print("\n" + "="*56)
        print(f"🔥 MEJORES PARLAYS MULTI-MERCADO (Top 4)")
        print("="*56)
        
        parlays_encontrados = sorted(parlays_encontrados, key=lambda x: x['ev'], reverse=True)
        
        for i, parlay in enumerate(parlays_encontrados[:4]):
            max_stake_pct = 0.04 if parlay['legs'] == 2 else 0.015 
            stake = min(max(0, (parlay['ev'] / (parlay['odd'] - 1)) * KELLY_FRACCIONAL), max_stake_pct)
            monto = bankroll * stake
            
            if monto > 0:
                inversion_total += monto
                detalle_ticket = " + ".join([f"{leg['partido']} ({leg['mercado']})" for leg in parlay['combo']])
                
                print(f"\n🎟️ TICKET #{i+1} ({parlay['legs']} Selecciones) | Cuota: {parlay['odd']:.2f}")
                for j, leg in enumerate(parlay['combo']):
                    print(f"   [{j+1}] {leg['partido']} -> {leg['mercado']} (@{leg['odd']:.2f})")
                print(f"   📈 EV: +{parlay['ev']*100:.1f}% | 💰 INVERTIR: €{monto:.2f}")

                log_apuestas_parlays.append({
                    'Fecha_Analisis': fecha_ejecucion,
                    'Tipo': f"Parlay {parlay['legs']}",
                    'Detalle_Selecciones': detalle_ticket,
                    'Cuota_Total': round(parlay['odd'], 2),
                    'Prob_IA': f"{parlay['prob']*100:.2f}%",
                    'EV': f"+{parlay['ev']*100:.2f}%",
                    'Inversion_Euros': round(monto, 2),
                    'Resultado_Real': ''
                })

    print("\n" + "="*56)
    print(f"💵 BANKROLL TOTAL A INVERTIR ESTA JORNADA: €{inversion_total:.2f}")
    print("="*56)

    # Exportación de apuestas al Excel (Guardando en la carpeta raíz)
    if len(log_apuestas_simples) > 0 or len(log_apuestas_parlays) > 0:
        print("\n📝 Guardando registros en Excel...")
        archivo_excel = os.path.join(BASE_DIR, 'Registro_Oraculo.xlsx')
        
        df_nuevas_simples = pd.DataFrame(log_apuestas_simples)
        df_nuevos_parlays = pd.DataFrame(log_apuestas_parlays)
        
        if os.path.exists(archivo_excel):
            try:
                df_hist_simples = pd.read_excel(archivo_excel, sheet_name='Simples')
                df_simples_final = pd.concat([df_hist_simples, df_nuevas_simples], ignore_index=True)
            except: df_simples_final = df_nuevas_simples
                
            try:
                df_hist_parlays = pd.read_excel(archivo_excel, sheet_name='Parlays')
                df_parlays_final = pd.concat([df_hist_parlays, df_nuevos_parlays], ignore_index=True)
            except: df_parlays_final = df_nuevos_parlays
        else:
            df_simples_final = df_nuevas_simples
            df_parlays_final = df_nuevos_parlays
            
        with pd.ExcelWriter(archivo_excel, engine='openpyxl') as writer:
            if not df_simples_final.empty:
                df_simples_final.to_excel(writer, sheet_name='Simples', index=False)
            if not df_parlays_final.empty:
                df_parlays_final.to_excel(writer, sheet_name='Parlays', index=False)
                
        print(f"✅ ¡Registro contable guardado con éxito en '{archivo_excel}'!")

# Arrancar simulaciones solo si cargó la base de datos
if not df_super.empty:
    oraculo_maestro_contable(partidos_jornada, df_super, MI_BANKROLL_TOTAL)