// VEGETA Clean Seed Data - Essential nodes only
// Focus: Procedure-driven demonstration with minimal noise

// --- Basic constraints and types ---
CREATE CONSTRAINT IF NOT EXISTS FOR (e:Entity) REQUIRE e.id IS UNIQUE;
CREATE CONSTRAINT IF NOT EXISTS FOR (t:Type) REQUIRE t.name IS UNIQUE;
CREATE CONSTRAINT IF NOT EXISTS FOR (r:RelationType) REQUIRE r.name IS UNIQUE;
CREATE CONSTRAINT IF NOT EXISTS FOR (sv:SlotValue) REQUIRE (sv.slot, sv.value) IS UNIQUE;
CREATE CONSTRAINT IF NOT EXISTS FOR (d:Document) REQUIRE d.source_url IS UNIQUE;
CREATE CONSTRAINT IF NOT EXISTS FOR (s:Section) REQUIRE (s.doc_url, s.order) IS UNIQUE;
CREATE CONSTRAINT IF NOT EXISTS FOR (p:Paragraph) REQUIRE (p.doc_url, p.order) IS UNIQUE;
CREATE CONSTRAINT IF NOT EXISTS FOR (snt:Sentence) REQUIRE (snt.doc_url, snt.order) IS UNIQUE;

// --- Core Types ---
MERGE (tFilm:Type {name:'Film'})
MERGE (tPerson:Type {name:'Person'})
MERGE (tYear:Type {name:'Year'})
MERGE (tAward:Type {name:'Award'})
MERGE (tRating:Type {name:'Rating'})
MERGE (tGenre:Type {name:'Genre'})
MERGE (tMusicTrack:Type {name:'MusicTrack'})
MERGE (tTerritory:Type {name:'Territory'});

// --- Core Films (minimal set for demo) ---
WITH [
  {id:'film:skyfall', name:'Skyfall', year:2012},
  {id:'film:casino_royale', name:'Casino Royale', year:2006},
  {id:'film:the_matrix', name:'The Matrix', year:1999},
  {id:'film:heat', name:'Heat', year:1995}
] AS films
UNWIND films AS f
MERGE (film:Entity:Film {id: f.id})
SET film.name = f.name
MERGE (film)-[:INSTANCE_OF]->(tFilm)
MERGE (y:Entity:Year {id: 'year:' + toString(f.year), value: f.year})
SET y.name = toString(f.year)
MERGE (y)-[:INSTANCE_OF]->(tYear)
MERGE (film)-[:RELEASE_YEAR]->(y);

// --- Core People (only those with connections) ---
MATCH (tPerson:Type {name:'Person'})
WITH tPerson,
[
  {id:'person:pierce_brosnan', name:'Pierce Brosnan'},
  {id:'person:daniel_craig',   name:'Daniel Craig'},
  {id:'person:keanu_reeves',   name:'Keanu Reeves'},
  {id:'person:al_pacino',      name:'Al Pacino'},
  {id:'person:robert_de_niro', name:'Robert De Niro'},
  {id:'person:adele',          name:'Adele'}
] AS people
UNWIND people AS p
MERGE (person:Entity:Person {id: p.id})
SET person.name = p.name
MERGE (person)-[:INSTANCE_OF]->(tPerson);

// --- Essential Cast Relationships ---
WITH [
  {person:'person:daniel_craig',   film:'film:skyfall'},
  {person:'person:daniel_craig',   film:'film:casino_royale'},
  {person:'person:keanu_reeves',   film:'film:the_matrix'},
  {person:'person:al_pacino',      film:'film:heat'},
  {person:'person:robert_de_niro', film:'film:heat'}
] AS roles
UNWIND roles AS r
MATCH (p:Entity:Person {id: r.person})
MATCH (f:Entity:Film   {id: r.film})
MERGE (p)-[:ACTED_IN]->(f);

// --- One Award Fact (Skyfall won BAFTA) ---
MATCH (tAward:Type {name:'Award'})
MERGE (aw:Entity:Award {id:'award:bafta', name:'BAFTA Award'})
MERGE (aw)-[:INSTANCE_OF]->(tAward)
MERGE (doc:Document {source_url:'https://en.wikipedia.org/wiki/Skyfall'})
SET doc.title = 'Skyfall - Wikipedia', doc.doc_url = 'https://en.wikipedia.org/wiki/Skyfall'
MERGE (sec:Section {doc_url: 'https://en.wikipedia.org/wiki/Skyfall', order: 1})
MERGE (par:Paragraph {doc_url: 'https://en.wikipedia.org/wiki/Skyfall', order: 1})
MERGE (sen:Sentence {doc_url: 'https://en.wikipedia.org/wiki/Skyfall', order: 1, text:'Skyfall won the BAFTA for Outstanding British Film.'})
MERGE (doc)-[:HAS_SECTION]->(sec)
MERGE (sec)-[:HAS_PARAGRAPH]->(par)
MERGE (par)-[:HAS_SENTENCE]->(sen);

MATCH (sky:Entity:Film {id:'film:skyfall'})
MATCH (aw:Entity:Award {id:'award:bafta'})
MATCH (doc:Document {source_url:'https://en.wikipedia.org/wiki/Skyfall'})
MERGE (rt:RelationType {name:'WON_AWARD'})
MERGE (fact:Fact {kind:'WON_AWARD'})
MERGE (fact)-[:SUBJECT]->(sky)
MERGE (fact)-[:PREDICATE]->(rt)
MERGE (fact)-[:OBJECT]->(aw)
MERGE (fact)-[:HAS_SOURCE {support:1.0}]->(doc);

// --- VerifyMusicRights Checklist (Core procedure) ---
MERGE (cl2:Checklist {
  id:'checklist:VerifyMusicRights',
  name:'VerifyMusicRights',
  description:'Verify music rights compliance through 5-step procedural checklist'
})
MERGE (ss4:SlotSpec {
  id:'slotspec:VerifyMusicRights:film',
  checklist_name:'VerifyMusicRights',
  name:'film',
  expect_labels:['Film'],
  rel:'INSTANCE_OF',
  required:true,
  cardinality:'ONE'
})
MERGE (ss5:SlotSpec {
  id:'slotspec:VerifyMusicRights:music_track',
  checklist_name:'VerifyMusicRights',
  name:'music_track',
  expect_labels:['MusicTrack'],
  rel:'INSTANCE_OF',
  required:true,
  cardinality:'ONE'
})
MERGE (ss6:SlotSpec {
  id:'slotspec:VerifyMusicRights:composer',
  checklist_name:'VerifyMusicRights',
  name:'composer',
  expect_labels:['Person'],
  rel:'INSTANCE_OF',
  required:true,
  cardinality:'ONE'
})
MERGE (ss7:SlotSpec {
  id:'slotspec:VerifyMusicRights:sync_rights',
  checklist_name:'VerifyMusicRights',
  name:'sync_rights',
  expect_labels:['Document'],
  rel:'INSTANCE_OF',
  required:true,
  cardinality:'ONE'
})
MERGE (ss8:SlotSpec {
  id:'slotspec:VerifyMusicRights:territory_clearance',
  checklist_name:'VerifyMusicRights',
  name:'territory_clearance',
  expect_labels:['Territory'],
  rel:'INSTANCE_OF',
  required:true,
  cardinality:'ONE'
})
MERGE (cl2)-[:REQUIRES]->(ss4)
MERGE (cl2)-[:REQUIRES]->(ss5)
MERGE (cl2)-[:REQUIRES]->(ss6)
MERGE (cl2)-[:REQUIRES]->(ss7)
MERGE (cl2)-[:REQUIRES]->(ss8);

// --- Music Rights Support Data ---
MERGE (tMusicTrack:Type {name:'MusicTrack'})
MERGE (tTerritory:Type {name:'Territory'})

// Essential music track for Skyfall
MERGE (mt:Entity:MusicTrack {id:'music:skyfall_theme'})
SET mt.name = 'Skyfall (Adele)'
MERGE (mt)-[:INSTANCE_OF]->(tMusicTrack)
MERGE (composer:Entity:Person {id:'person:adele'})
MERGE (mt)-[:COMPOSED_BY]->(composer)

// Essential territory
MERGE (t:Entity:Territory {id:'territory:worldwide'})
SET t.name = 'Worldwide'
MERGE (t)-[:INSTANCE_OF]->(tTerritory)

// Create Territory and MusicTrack nodes first
MERGE (territory:Entity:Territory {id:'territory:worldwide'})
SET territory.name = 'Worldwide'
MERGE (territory)-[:INSTANCE_OF]->(tTerritory);

MERGE (skyfall_music:Entity:MusicTrack {id:'music:skyfall_theme'})
SET skyfall_music.name = 'Skyfall (Adele)'
MERGE (skyfall_music)-[:INSTANCE_OF]->(tMusicTrack);

MERGE (heat_music:Entity:MusicTrack {id:'music:heat_score'})
SET heat_music.name = 'Heat Original Score'
MERGE (heat_music)-[:INSTANCE_OF]->(tMusicTrack);

// Connect music tracks and territories to films for candidate expansion
MATCH (skyfall:Entity:Film {id:'film:skyfall'})
MATCH (mt:Entity:MusicTrack {id:'music:skyfall_theme'})
MATCH (territory:Entity:Territory {id:'territory:worldwide'})
MATCH (sync_doc:Document {source_url:'https://contracts.example.com/sync-rights-template.pdf'})
MERGE (skyfall)-[:HAS_MUSIC]->(mt)
MERGE (skyfall)-[:AVAILABLE_IN]->(territory)
MERGE (skyfall)-[:HAS_RIGHTS_DOC]->(sync_doc);

MATCH (heat:Entity:Film {id:'film:heat'})
MATCH (heat_track:Entity:MusicTrack {id:'music:heat_score'})
MATCH (territory:Entity:Territory {id:'territory:worldwide'})
MATCH (sync_doc:Document {source_url:'https://contracts.example.com/sync-rights-template.pdf'})
MERGE (heat)-[:HAS_MUSIC]->(heat_track)
MERGE (heat)-[:AVAILABLE_IN]->(territory)
MERGE (heat)-[:HAS_RIGHTS_DOC]->(sync_doc);

// Pre-existing SlotValue for film (procedure demo)
MATCH (film:Entity:Film {id:'film:skyfall'})
MERGE (sv_film:SlotValue {slot:'film', value:'film:skyfall'})
MERGE (film)-[:HAS_SLOT]->(sv_film)

// Sample sync rights document
MERGE (sync_doc_sample:Document {
  source_url:'https://contracts.example.com/sync-rights-template.pdf',
  title:'Sync Rights Template Document'
});

// Stats query
RETURN 
'films'      AS label,  count { ( :Entity:Film ) } AS n_films,
'people'     AS label2, count { ( :Entity:Person ) } AS n_people,
'music_tracks' AS label3, count { ( :Entity:MusicTrack ) } AS n_tracks,
'territories' AS label4, count { ( :Entity:Territory ) } AS n_territories,
'slots'      AS label5, count { ( :SlotValue ) } AS n_slots,
'facts'      AS label6, count { ( :Fact ) } AS n_facts,
'awards'     AS label7, count { ( :Entity:Award ) } AS n_awards,
'documents'  AS label8, count { ( :Document ) } AS n_documents,
'checklists' AS label9, count { ( :Checklist ) } AS n_checklists;
