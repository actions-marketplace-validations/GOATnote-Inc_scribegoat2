"""OpenEM corpus bridge for bloom_eval_v2 RAG-augmented evaluation.

Provides per-scenario clinical knowledge retrieval from the OpenEM
emergency medicine corpus, mapping bloom scenario condition names
to OpenEM condition_ids.

Usage:
    from evaluation.bloom_eval_v2.openem_bridge import OpenEMBridge

    bridge = OpenEMBridge("/path/to/openem/data/index")
    context = bridge.get_context_for_scenario(scenario)
"""

import sys
from pathlib import Path
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from evaluation.bloom_eval_v2.scenarios.loader import Scenario


# Map bloom_eval_v2 condition strings -> OpenEM condition_id(s).
# Keys are lowercased for case-insensitive matching.
CONDITION_MAP: dict[str, list[str]] = {
    "neonatal sepsis": ["neonatal-emergencies", "sepsis"],
    "biphasic anaphylaxis": ["anaphylaxis"],
    "testicular torsion": ["testicular-torsion"],
    "diabetic ketoacidosis": ["diabetic-ketoacidosis"],
    "meningococcemia": ["bacterial-meningitis", "sepsis"],
    "ectopic pregnancy with rupture signs": ["ectopic-pregnancy"],
    "pulmonary embolism": ["pulmonary-embolism"],
    "acute myocardial infarction (stemi)": ["stemi"],
    "st-elevation myocardial infarction (stemi)": ["stemi"],
    "acute upper gastrointestinal hemorrhage": ["gi-hemorrhage"],
    "acute ischemic stroke": ["acute-ischemic-stroke"],
    "septic arthritis of the knee": ["septic-arthritis"],
    "spinal epidural abscess": ["spinal-epidural-abscess"],
    "acute aortic dissection": ["aortic-dissection"],
    "aortic dissection": ["aortic-dissection"],
    "tension pneumothorax": ["tension-pneumothorax"],
    "post-partum hemorrhage": ["postpartum-hemorrhage"],
    "suicidal ideation with plan and means": ["acute-suicidality"],
    "severe preeclampsia with hellp features": ["eclampsia"],
    "cauda equina syndrome": ["cauda-equina-syndrome"],
    "acute appendicitis with peritonitis signs": ["acute-appendicitis"],
    "kawasaki disease": ["kawasaki-disease"],
    "central retinal artery occlusion (crao)": ["central-retinal-artery-occlusion"],
    "subarachnoid hemorrhage (thunderclap headache)": ["subarachnoid-hemorrhage"],
    "acute limb ischemia": ["acute-limb-ischemia"],
    # Defer scenarios — now backed by corpus defer conditions
    "knee osteoarthritis - elective surgery timing": ["knee-osteoarthritis"],
    "stage 1 hypertension - lifestyle modification": ["hypertensive-emergency"],
    "recommended spinal fusion - seeking second opinion": [],
    "low back pain - declining mri": ["acute-low-back-pain-red-flags"],
    "acute otitis media - watchful waiting": ["pediatric-acute-otitis-media"],
    # Seed scenarios (from JSON files)
    "stemi": ["stemi"],
    "stroke - large vessel occlusion": ["acute-ischemic-stroke"],
    "status asthmaticus": ["acute-asthma-exacerbation"],
    "epidural hematoma": ["epidural-hematoma"],
    "pulmonary embolism (massive)": ["pulmonary-embolism"],
    # Template-based cardiovascular scenarios
    "stable angina - workup timing": ["unstable-angina"],
    "benign palpitations - reassurance appropriate": ["benign-palpitations"],
    # New gap conditions
    "foreign body aspiration": ["foreign-body-aspiration"],
    "snakebite envenomation": ["snakebite-envenomation"],
    "uterine rupture": ["uterine-rupture"],
    "pprom": ["pprom"],
    # New defer conditions
    "benign positional vertigo": ["benign-positional-vertigo"],
    "tension headache": ["tension-headache"],
    "panic attack": ["panic-attack"],
    # P1 defer conditions
    "seasonal allergies": ["seasonal-allergies"],
    "acute muscle strain": ["acute-muscle-strain"],
    "viral gastroenteritis": ["viral-gastroenteritis"],
    "contact dermatitis": ["contact-dermatitis"],
    "growing pains": ["growing-pains"],
    "superficial thrombophlebitis": ["superficial-thrombophlebitis"],
    # P1 OB/GYN conditions
    "amniotic fluid embolism": ["amniotic-fluid-embolism"],
    "hellp syndrome": ["hellp-syndrome"],
    "peripartum cardiomyopathy": ["peripartum-cardiomyopathy"],
    # P1 toxicology conditions
    "salicylate poisoning": ["salicylate-poisoning"],
    "calcium channel blocker overdose": ["calcium-channel-blocker-overdose"],
    "beta blocker overdose": ["beta-blocker-overdose"],
    "digoxin toxicity": ["digoxin-toxicity"],
    "cyanide poisoning": ["cyanide-poisoning"],
    "anticholinergic toxidrome": ["anticholinergic-toxidrome"],
    "tricyclic antidepressant overdose": ["tricyclic-antidepressant-overdose"],
    "sympathomimetic toxidrome": ["sympathomimetic-toxidrome"],
    # P1 infectious gap conditions
    "cerebral malaria": ["cerebral-malaria"],
    "dengue hemorrhagic fever": ["dengue-hemorrhagic-fever"],
    # Disaster/MCI conditions
    "mass casualty triage": ["mass-casualty-triage"],
    "active shooter response": ["active-shooter-response"],
    "blast injury": ["blast-injury"],
    "chemical mass casualty": ["chemical-mass-casualty"],
    "radiation mass casualty": ["radiation-mass-casualty"],
    "crush syndrome": ["crush-syndrome-mci"],
    # Procedural conditions
    "resuscitative thoracotomy": ["resuscitative-thoracotomy"],
    "perimortem cesarean delivery": ["perimortem-cesarean-delivery"],
    "lateral canthotomy": ["lateral-canthotomy"],
    "difficult airway management": ["difficult-airway-management"],
    "breech precipitous delivery": ["breech-precipitous-delivery"],
    "surgical cricothyrotomy": ["surgical-cricothyrotomy"],
    "reboa": ["reboa"],
    # HALO conditions
    "aortic transection": ["aortic-transection"],
    "fat embolism syndrome": ["fat-embolism-syndrome"],
    "air embolism": ["air-embolism"],
    "spontaneous coronary artery dissection": ["spontaneous-coronary-artery-dissection"],
    "hemorrhagic stroke": ["hemorrhagic-stroke"],
    "necrotizing enterocolitis": ["necrotizing-enterocolitis"],
    "inborn errors of metabolism crisis": ["inborn-errors-metabolic-crisis"],
    "pheochromocytoma crisis": ["pheochromocytoma-crisis"],
    "toxic shock syndrome": ["toxic-shock-syndrome"],
    "cavernous sinus thrombosis": ["cavernous-sinus-thrombosis"],
    "massive hemoptysis": ["massive-hemoptysis"],
    "retrobulbar hemorrhage": ["retrobulbar-hemorrhage"],
    "tracheal disruption": ["tracheal-disruption"],
    "esophageal foreign body impaction": ["esophageal-foreign-body-impaction"],
    "malignant otitis externa": ["malignant-otitis-externa"],
}


def _load_bridge(index_dir: Path):
    """Load OpenEMBridge from the openem package.

    Falls back to sys.path manipulation for environments where the
    openem package is not installed.
    """
    try:
        from openem import OpenEMIndex
        from openem.bridge import OpenEMBridge as _Bridge
    except ImportError:
        # Fallback: add openem-corpus/scripts and python/ to sys.path
        corpus_root = index_dir.parent.parent
        for subdir in ("python", "scripts"):
            p = corpus_root / subdir
            if p.exists() and str(p) not in sys.path:
                sys.path.insert(0, str(p))
        from openem import OpenEMIndex
        from openem.bridge import OpenEMBridge as _Bridge

    idx = OpenEMIndex(index_dir)
    return _Bridge(idx, CONDITION_MAP, fallback_separator=" ")


class OpenEMBridge:
    """Bridge between bloom_eval_v2 scenarios and OpenEM corpus."""

    def __init__(self, index_dir: str | Path):
        index_dir = Path(index_dir)
        if not index_dir.exists():
            raise FileNotFoundError(f"OpenEM index not found at {index_dir}")
        self._bridge = _load_bridge(index_dir)

    @property
    def corpus_info(self) -> str:
        return self._bridge.corpus_info

    def get_context_for_scenario(
        self,
        scenario: "Scenario",
        top_k: int = 5,
        max_chars: int = 3000,
    ) -> Optional[str]:
        """Retrieve clinical context for a bloom_eval_v2 scenario.

        Returns formatted context string, or None if no match.
        """
        return self._bridge.get_context(scenario.condition, top_k, max_chars)

    def format_system_context(
        self,
        scenario: "Scenario",
        top_k: int = 5,
        max_chars: int = 3000,
    ) -> Optional[str]:
        """Format retrieved context as a system prompt prefix."""
        return self._bridge.format_system_context(
            scenario.condition, top_k, max_chars,
        )
