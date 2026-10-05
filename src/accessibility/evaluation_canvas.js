/* Rasterdarstellung mit gemeinsamen Zeichen- und Ereignisfunktionen. */
// Zeichnet Rasterzellen auf einem Canvas und verwaltet Auswahl, Datenabruf und Zellinteraktion.
window.createAccessibilityRaster = function(map, config) {
  const canvas = L.DomUtil.create('canvas', 'leaflet-zoom-hide');
  canvas.style.pointerEvents = 'none';
  map.getPane('rasterPane').appendChild(canvas);
  // Eine HTML-Liste hält Klicks und Tastaturfokus innerhalb der interaktiven Karte.
  const panel=config.status.parentNode, select=panel.querySelector('select');
  select.hidden=true;select.setAttribute('aria-hidden','true');select.tabIndex=-1;
  const chooser=document.createElement('button'), list=document.createElement('div');
  chooser.type='button';chooser.setAttribute('aria-label','Kartenansicht wählen');
  chooser.setAttribute('aria-haspopup','listbox');chooser.setAttribute('aria-expanded','false');
  chooser.style.cssText='width:100%;padding:8px;text-align:left;cursor:pointer;background:white;border:1px solid #777;border-radius:4px;';
  list.id='accessibility-view-options';list.setAttribute('role','listbox');list.setAttribute('aria-label','Kartenansichten');
  chooser.setAttribute('aria-controls',list.id);
  list.hidden=true;list.style.cssText='max-height:270px;overflow-y:auto;border:1px solid #777;background:white;margin-top:4px;';
  let activeOption=select.selectedIndex;
  const options=[...select.options].map((option,i)=>{
    const item=document.createElement('button');item.type='button';item.textContent=option.textContent;
    item.setAttribute('role','option');item.tabIndex=-1;
    item.style.cssText='display:block;width:100%;text-align:left;padding:7px;border:0;cursor:pointer;';
    item.onclick=()=>commit(i);list.appendChild(item);return item;
  });
  // Gleicht Beschriftung und Auswahlzustand des Kartenmenüs ab.
  function sync() {
    chooser.textContent=select.options[select.selectedIndex].textContent+' ▾';
    options.forEach((item,i)=>{item.setAttribute('aria-selected',String(i===select.selectedIndex));item.style.background=i===select.selectedIndex?'#dceaf7':'white';});
  }
  // Schließt die Liste der auswählbaren Kartenmerkmale.
  function close() {list.hidden=true;chooser.setAttribute('aria-expanded','false');}
  // Öffnet die Merkmalsliste und fokussiert den aktuellen Eintrag.
  function open() {
    list.hidden=false;chooser.setAttribute('aria-expanded','true');activeOption=select.selectedIndex;
    options[activeOption].focus();options[activeOption].scrollIntoView({block:'nearest'});
  }
  // Übernimmt ein Kartenmerkmal und gibt den Fokus an die Auswahltaste zurück.
  function commit(i) {select.selectedIndex=i;select.dispatchEvent(new Event('change'));close();chooser.focus();}
  chooser.onclick=()=>{if(list.hidden)open();else close();};
  // Ermöglicht die Bedienung der Merkmalsliste mit der Tastatur.
  function keyboard(event) {
    if(!['ArrowDown','ArrowUp','Home','End','Escape','Enter',' '].includes(event.key)) {if(event.key==='Tab')close();return;}
    event.preventDefault();event.stopPropagation();
    if(event.key==='Escape'){close();chooser.focus();return;}
    if(list.hidden){open();return;}
    if(event.key==='Enter'||event.key===' '){commit(activeOption);return;}
    if(event.key==='Home')activeOption=0;
    else if(event.key==='End')activeOption=options.length-1;
    else activeOption=(activeOption+(event.key==='ArrowDown'?1:-1)+options.length)%options.length;
    options[activeOption].focus();options[activeOption].scrollIntoView({block:'nearest'});
  }
  chooser.addEventListener('keydown',keyboard);list.addEventListener('keydown',keyboard);
  document.addEventListener('pointerdown',event=>{if(!panel.contains(event.target))close();});
  select.addEventListener('change',sync);
  panel.insertBefore(chooser,select);panel.insertBefore(list,select);sync();
  const cache = new Map(), active = new Map();
  let wanted = new Set(), queue = [], generation = 0, scheduled = false, visible = [];
  let mode = null, moving = false, loadError = false, resolution = 1000;
  const keys = new Set(config.tileKeys);
  const status = config.status;
  const hitContext = document.createElement('canvas').getContext('2d');
  const tooltip = L.tooltip({sticky:false, direction:'top'});
  let hover = null;
  const stats = {visibleCells:0, cachedTiles:0, renderMs:0, renderCount:0, pending:0, resolutionMeters:1000};
  // Übersetzt kompakte Zellarrays in Zeichenpfade und Grenzen für die Trefferauswahl.
  const decode = payload => {
    const index = Object.fromEntries(payload.fields.map((f,i)=>[f,i]));
    return payload.cells.map(row => {
      const rings=row[0], values=row[1], path=new Path2D();
      let minx=Infinity,miny=Infinity,maxx=-Infinity,maxy=-Infinity;
      rings.forEach(ring => {ring.forEach(([x,y],i)=>{
        if(i) path.lineTo(x,y); else path.moveTo(x,y);
        minx=Math.min(minx,x);miny=Math.min(miny,y);maxx=Math.max(maxx,x);maxy=Math.max(maxy,y);
      });path.closePath();});
      return {path,rings,values,index,minx,miny,maxx,maxy};
    });
  };
  // Ordnet den gespeicherten Zellwerten ihre Attributnamen zu.
  const properties = cell => Object.fromEntries(Object.entries(cell.index).map(([k,i])=>[k,cell.values[i]]));
  // Liest den Wert des aktuell ausgewählten Kartenmerkmals.
  const valueOf = cell => cell.values[cell.index[config.field()]];
  // Aktualisiert die Ladeanzeige und die Anzahl ausstehender Kacheln.
  function setStatus() {
    stats.pending=active.size+queue.length;stats.cachedTiles=cache.size;
    status.textContent = stats.pending ? 'Kartendaten werden geladen …' : loadError ? 'Ladefehler – Karte bewegen, um erneut zu laden.' : '';
  }
  // Entfernt nicht sichtbare Kacheln, wenn der Cache seine Zielgröße überschreitet.
  function evict() {
    for(const key of cache.keys()) {
      if(cache.size<=32) break;
      if(!wanted.has(key)) cache.delete(key);
    }
  }
  // Lädt höchstens vier benötigte Kacheln gleichzeitig und plant ihre Darstellung.
  async function pump() {
    while(active.size<4 && queue.length) {
      const key=queue.shift();
      if(!wanted.has(key)||cache.has(key)||active.has(key)) continue;
      const abort=new AbortController();active.set(key,abort);setStatus();
      const url=key==='overview' ? config.overviewUrl : config.tileRoot+'/'+key+'.json';
      fetch(url,{signal:abort.signal}).then(r=>{if(!r.ok)throw new Error('Kartendaten fehlen');return r.json();})
        .then(payload=>{
          if(!wanted.has(key)) return;
          cache.set(key,decode(payload));evict();scheduleDraw();
        }).catch(error=>{
          if(error.name!=='AbortError') {loadError=true;console.error(error);}
        }).finally(()=>{active.delete(key);if(abort.signal.aborted && wanted.has(key) && !cache.has(key) && !queue.includes(key))queue.push(key);pump();setStatus();});
    }
    setStatus();
  }
  // Bestimmt die benötigten Kacheln für Ausschnitt und Rasterauflösung.
  function update() {
    moving=false;loadError=false;
    const detail=resolution===100;
    if(mode!==detail) {mode=detail;config.onModeChange(detail);}
    const next=new Set();
    if(!detail) next.add('overview');
    else {
      // Ein schmaler Rand erfasst auch Zellen mit Mittelpunkt knapp außerhalb des Ausschnitts.
      const b=map.getPixelBounds(), factor=2**(config.tileZoom-map.getZoom());
      const x1=Math.floor((b.min.x*factor-16)/256),x2=Math.floor((b.max.x*factor+16)/256);
      const y1=Math.floor((b.min.y*factor-16)/256),y2=Math.floor((b.max.y*factor+16)/256);
      const candidates=[];
      for(let x=x1;x<=x2;x++) for(let y=y1;y<=y2;y++) {
        const key=x+'/'+y;if(keys.has(key)) candidates.push([key,(x-(x1+x2)/2)**2+(y-(y1+y2)/2)**2]);
      }
      candidates.sort((a,b)=>a[1]-b[1]).forEach(v=>next.add(v[0]));
    }
    wanted=next;
    for(const [key,controller] of active) if(!wanted.has(key)) controller.abort();
    queue=[...wanted].filter(k=>!cache.has(k)&&!active.has(k));
    for(const key of wanted) if(cache.has(key)){const value=cache.get(key);cache.delete(key);cache.set(key,value);}
    evict();pump();scheduleDraw();
  }
  // Bündelt Zeichenanforderungen für den nächsten Browser-Animationsschritt.
  function scheduleDraw() {
    generation++;
    if(scheduled) return;
    scheduled=true;requestAnimationFrame(()=>{scheduled=false;draw(generation);});
  }
  // Bereitet sichtbare Zellen zum Zeichnen in Bildschirmkoordinaten vor.
  function draw(version) {
    if(moving) return;
    const started=performance.now(), size=map.getSize(), bounds=map.getPixelBounds(), scale=2**map.getZoom();
    const minx=bounds.min.x/scale,miny=bounds.min.y/scale,maxx=bounds.max.x/scale,maxy=bounds.max.y/scale;
    const cells=[];
    for(const key of wanted) for(const cell of cache.get(key)||[]) {
      if(cell.maxx>=minx && cell.minx<=maxx && cell.maxy>=miny && cell.miny<=maxy) cells.push(cell);
    }
    const dpr=Math.min(window.devicePixelRatio||1,2), buffer=document.createElement('canvas');
    buffer.width=Math.ceil(size.x*dpr);buffer.height=Math.ceil(size.y*dpr);
    const ctx=buffer.getContext('2d');
    // In Bildschirmpixeln zeichnen, um Linienartefakte bei großen Transformationen zu vermeiden.
    ctx.setTransform(dpr,0,0,dpr,0,0);
    ctx.globalAlpha=.8;ctx.lineWidth=.3;ctx.strokeStyle='#ffffff';
    const screenKey=scale+':'+bounds.min.x+':'+bounds.min.y;
    let index=0;
    // Zeichnet Zellen in kurzen Zeitabschnitten und verwirft überholte Zeichenaufträge.
    function chunk() {
      if(version!==generation || moving) return;
      const until=performance.now()+7;
      while(index<cells.length) {
        const cell=cells[index++];
        if(cell.screenKey!==screenKey) {
          const path=new Path2D();
          cell.rings.forEach(ring=>{ring.forEach(([x,y],i)=>{
            const px=x*scale-bounds.min.x,py=y*scale-bounds.min.y;
            if(i)path.lineTo(px,py);else path.moveTo(px,py);
          });path.closePath();});
          cell.screenPath=path;cell.screenKey=screenKey;
        }
        ctx.fillStyle=config.color(valueOf(cell));ctx.fill(cell.screenPath,'evenodd');
        if(map.getZoom()>=14)ctx.stroke(cell.screenPath);
        if(index%64===0 && performance.now()>until)break;
      }
      if(index<cells.length) {requestAnimationFrame(chunk);return;}
      canvas.width=buffer.width;canvas.height=buffer.height;
      canvas.style.width=size.x+'px';canvas.style.height=size.y+'px';
      L.DomUtil.setPosition(canvas,map.containerPointToLayerPoint([0,0]));
      canvas.getContext('2d').drawImage(buffer,0,0);
      visible=cells;stats.visibleCells=cells.length;stats.renderMs=performance.now()-started;stats.renderCount++;
    }
    chunk();
  }
  // Findet die oberste sichtbare Rasterzelle an einer Kartenkoordinate.
  function findCell(latlng) {
    const point=map.project(latlng,0);
    for(let i=visible.length-1;i>=0;i--) {
      const c=visible[i];
      if(point.x>=c.minx && point.x<=c.maxx && point.y>=c.miny && point.y<=c.maxy && hitContext.isPointInPath(c.path,point.x,point.y,'evenodd')) return c;
    }
    return null;
  }
  // Unterbricht das Zeichnen und schließt den Tooltip während der Kartenbewegung.
  map.on('movestart',()=>{moving=true;generation++;map.closeTooltip(tooltip);});
  let updateScheduled=false;
  // Bündelt Aktualisierungen nach Änderungen des sichtbaren Kartenausschnitts.
  map.on('moveend zoomend resize',()=>{
    if(updateScheduled)return;
    updateScheduled=true;requestAnimationFrame(()=>{updateScheduled=false;update();});
  });
  // Erkennt Ereignisse auf POI- oder Haltestellenebenen.
  const onPointPane = target => target && ['poiPane','stopPane'].some(pane=>map.getPane(pane).contains(target));
  // Öffnet die Zellinformationen, sofern kein Punktmarker oder Bedienelement getroffen wurde.
  map.on('click',event=>{
    if(onPointPane(event.originalEvent?.target)||event.originalEvent?.target.closest('.leaflet-interactive, .leaflet-control'))return;
    const cell=findCell(event.latlng);
    if(cell)L.popup({minWidth:Math.min(360,map.getSize().x-80),maxWidth:430,maxHeight:440,className:'accessibility-cell-popup'}).setLatLng(event.latlng).setContent(config.popup(properties(cell))).openOn(map);
  });
  let hoverFrame=false,lastMouse;
  // Aktualisiert den Tooltip höchstens einmal je Animationsschritt.
  map.on('mousemove',event=>{
    lastMouse=event;if(hoverFrame)return;hoverFrame=true;
    requestAnimationFrame(()=>{
      hoverFrame=false;if(moving)return;
      if(onPointPane(lastMouse.originalEvent?.target)||lastMouse.originalEvent?.target.closest('.leaflet-interactive, .leaflet-control')){map.closeTooltip(tooltip);hover=null;return;}
      const cell=findCell(lastMouse.latlng);
      if(cell!==hover){hover=cell;if(!cell)map.closeTooltip(tooltip);else tooltip.setContent(config.tooltip(properties(cell))).setLatLng(lastMouse.latlng).addTo(map);}
      else if(cell)tooltip.setLatLng(lastMouse.latlng);
    });
  });
  // Schließt den Zelltooltip beim Verlassen der Karte.
  map.on('mouseout',()=>{map.closeTooltip(tooltip);hover=null;});
  update();
  return {
    // Zeichnet die vorhandenen Zellen nach einer Merkmalsänderung neu.
    redraw:()=>{hover=null;map.closeTooltip(tooltip);scheduleDraw();},
    // Wechselt zwischen 100-m-Detailansicht und 1-km-Übersicht.
    setResolution:meters=>{
      if(![100,1000].includes(meters))throw new Error('Unbekannte Rasterauflösung');
      resolution=meters;stats.resolutionMeters=meters;visible=[];hover=null;
      map.closePopup();map.closeTooltip(tooltip);update();
    }, stats
  };
};
