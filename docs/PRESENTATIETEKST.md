# Presentatietekst — verdediging researchproject

Spreektekst bij de presentatiewebsite (`presentation/index.html`). Ongeveer 1100 woorden: met de demo erbij ± 9 minuten.
Tussen [haakjes]: wat je doet of welke toets je indrukt. Tijden: waar je ongeveer moet zitten.
Doel: rond **8:00** bij hoofdstuk 14 (het antwoord), want dan geeft de jury een seintje.

---

## 1 · VRAAG (0:00)

Drones die pakjes leveren of autonoom landen, vliegen met GPS tot boven hun landingsplatform. Maar GPS is maar tot op een paar meter nauwkeurig. De laatste meters moet de drone het platform dus zelf zien, met een camera en een kleine boordcomputer.

Mijn onderzoeksvraag is: hoe kan een TensorRT-geoptimaliseerde objectdetector op een NVIDIA Jetson Orin Nano een landingsplatform in realtime, dus aan minstens 20 beelden per seconde, detecteren, zodat de drone vanuit verschillende hoeken en afstanden correct beslist om te dalen of de landing af te breken?

## 2 · SIMULATIE (0:45)

[→] Echte drones laten crashen is duur, dus alles gebeurt in AirSim, een dronesimulator van Microsoft op Unreal Engine. Onder de drone hangt een camera die recht naar beneden kijkt, 640 op 640 pixels. Het platform is één meter op één meter, met een H zoals op een helikopterplatform.

## 3 · OPSTELLING (1:10)

[→] De simulator en de landingslogica draaien op mijn laptop. Elk camerabeeld gaat als JPEG over een USB-C-kabel naar de Jetson. Die zoekt het platform en stuurt de box en de zekerheid terug. Ik laat de Jetson bewust alléén de detectie doen: zo meet ik precies wat de edge-hardware bijdraagt.

## 4 · HOEK EN AFSTAND (1:40)

[→] Ik test vanuit verschillende hoeken, van recht erboven tot 45 graden, en op afstanden van 5 tot 30 meter. Verder dan 45 graden kan niet: dan valt het platform buiten beeld. Kijk naar de beelden: op 30 meter is het platform nog maar 11 pixels breed. Daar wordt het moeilijk.

## 5 · DATASET (2:05)

[→] Om een model te trainen heb je gelabelde beelden nodig. Met de hand zou dat dagen duren. Mijn script teleporteert de drone naar een positie, neemt een foto en vraagt aan AirSim zelf waar het platform in beeld staat. Zo had ik 4800 beelden, volledig automatisch gelabeld. De testbeelden komen van andere posities dan de trainingsbeelden, zodat ik eerlijk meet.

## 6 · ENGINES (2:35)

[→] Ik vergelijk twee modellen: YOLO26s, een klassiek convolutioneel netwerk, en RT-DETR-l, een transformer. Elk model zet ik om naar een TensorRT-engine in drie rekenprecisies: FP32, FP16 en INT8. Dat zijn zes engines, allemaal op de Jetson zelf gebouwd. Minder bits betekent sneller: FP16 halveert de rekentijd van YOLO26s, van 19 naar 9,4 milliseconden. INT8 haalt er nog bijna 30 procent af. RT-DETR-l is in elke vorm veel trager.

## 7 · 20 FPS (3:10)

[→] Waarom precies 20 beelden per seconde? De drone daalt met één meter per seconde en moet het platform vijf keer na elkaar zien. In die tijd mag hij maximaal 25 centimeter zakken. Vijf keer één, gedeeld door 0,25, is 20. [schuif v naar 2] Daal ik twee keer zo snel, dan wordt de eis 40, en haalt geen enkele engine het nog.

## 8 · KETEN (3:45)

[→] Dan de ketentest: de drone hangt stil, en ik stuur 500 beelden door de volledige keten, van camera tot beslissing. Deze balk is één beeld. Het grootste stuk is het beeld ophalen uit de simulator. Het netwerk is dat dunne streepje: één milliseconde. YOLO26s in FP16 blijft onder de 50 milliseconden: 22,6 beelden per seconde. [klik RT-DETR-l FP32] RT-DETR-l in FP32 heeft alleen al 65 milliseconden nodig om te rekenen. Die haalt de eis nooit.

## 9 · LANDINGSREGEL (4:25)

[→] De regel is eenvoudig. Zie ik het platform vijf keer op rij met minstens 50 procent zekerheid, dan daal ik. Mis ik het drie keer tijdens het dalen, dan breek ik af en blijft de drone hangen. Op deze grafiek zie je de zekerheid tijdens het dalen: altijd ruim boven de drempel.

## 10 · DEMO (4:55)

[0, wissel naar het demovenster] Nu live. Linksboven ziet u welke engine draait, rechtsboven de beelden per seconde.

[spatie] De drone start op 20 meter. Kijk naar de teller: bij vijf op vijf begint hij te dalen… en hij landt op het platform.

[p, spatie] Nu haal ik het platform weg. De drone ziet niets, wacht, en breekt af. Dat is precies wat hij moet doen.

[als er tijd is: 2] Met RT-DETR-l ziet u de beelden per seconde zakken.

[lukt het live niet: terug naar de website, V] Ik toon de opname van dezelfde demo.

## 11 · LIVE LANDINGEN (6:30)

[terug naar de website, →] Elke engine heeft 50 keer geland, samen 300 landingen. Alle 300 zijn correct beslist: 240 goede landingen en 60 keer terecht afgebroken. Geen enkele drone daalde zonder platform. 50 op 50 is natuurlijk geen bewijs van 100 procent: statistisch is het minstens 93 procent. En omdat alles perfect scoort, zie je hier nog géén verschil tussen de rekenprecisies.

## 12 · SLECHT ZICHT (7:00)

[→] Daarom heb ik het moeilijker gemaakt: mist, stof, weinig licht, onscherpte en ruis. [klik mist 0,5] Bij mist is FP16 even goed als FP32, maar INT8 mist het platform 14 procentpunten vaker. Kwantisatie kost dus pas iets als het moeilijk wordt. YOLO26s faalt wel veilig: ziet het niets, dan breekt de drone af.

## 13 · VALS PLATFORM (7:35)

[→] Dit is het gevaarlijkste resultaat. In mist ziet RT-DETR-l hier rechtsonder een platform dat er niet is, met 93 procent zekerheid. Een drone zou daar landen. Geen enkele drempel houdt dat tegen.

## 14 · ANTWOORD (7:55)

[→] Mijn antwoord op de onderzoeksvraag: YOLO26s in FP16.

- Het is snel genoeg: 22,6 beelden per seconde in de ketentest.
- Het heeft in de 300 landingen nooit fout beslist.
- Het is even nauwkeurig als FP32, ook bij slecht zicht.
- En het faalt veilig.

INT8 lijkt aantrekkelijk omdat het sneller rekent, maar in de keten wint het niets (22,0 tegenover 22,6 beelden per seconde), want de simulator neemt de meeste tijd. Bij slecht zicht verliest het herkenning. RT-DETR-l haalt de snelheid niet en ziet in mist platformen die er niet zijn.

## 15 · TOUCHDOWN (8:40)

[→] Tot slot de beperkingen, want die horen erbij.

- Tijdens het vliegen haal ik live maximaal 19,7 beelden per seconde, net onder de eis. Dat komt door de simulator die het beeld trager aflevert, niet door de Jetson.
- Alles is simulatie: echte beelden, wind en licht zijn de volgende stap.
- De laatste drie meter landt de drone blind. Zo dicht bij het platform valt de schaduw van de drone erover, en het model kreeg alleen beelden vanaf 4 meter te zien.

Bedankt voor uw aandacht. Zijn er nog vragen?

---

## Als je achterloopt

- **Na 4:30 nog niet bij de demo:** hoofdstuk 5 (dataset) en 9 (landingsregel) in één zin.
- **Na 7:30 nog niet bij hoofdstuk 12:** sla 13 (vals platform) over en zeg het in één zin bij 12: "RT-DETR-l ziet in mist zelfs platformen die er niet zijn."
- **Demo loopt mis:** niet proberen te herstellen, meteen V.
