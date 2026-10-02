import sys,subprocess,time,os
from playwright.sync_api import sync_playwright
os.chdir(os.path.dirname(os.path.abspath(__file__)))
srv=subprocess.Popen([sys.executable,'-m','http.server','8765'],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL); time.sleep(1)
jobs=[a.split('|') for a in sys.argv[1:]]  # out|query
try:
  with sync_playwright() as p:
    b=p.chromium.launch(executable_path='/opt/pw-browsers/chromium-1194/chrome-linux/chrome',args=['--use-gl=angle','--use-angle=swiftshader','--enable-unsafe-swiftshader'])
    pg=b.new_page(viewport={'width':1000,'height':700})
    for out,qs in jobs:
        pg.goto(f'http://localhost:8765/web/view.html?{qs}'); pg.wait_for_function("document.title=='ready'",timeout=120000)
        pg.locator('canvas').screenshot(path=out); print('ok',out)
    b.close()
finally: srv.terminate()
