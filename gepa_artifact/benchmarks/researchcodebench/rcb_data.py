from ..benchmark import Benchmark
import dspy

import json
from typing import Dict, Any, Iterable, List, Optional


def read_jsonl(path: str) -> Iterable[Dict[str, Any]]:
    """Yield one JSON object per line from a JSONL file."""
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)

def load_tasks(path: str) -> List[Dict[str, Any]]:
    """Load all records from a JSONL file into a list."""
    return list(read_jsonl(path))


class RCBenchmark(Benchmark):
    def init_dataset(self, with_gold=False):
        # Hardcoded instance ID splits
        train_instance_ids = ['TabDiff::model.py::123:125', 'DiffusionDPO::loss.py::64:66', 'DyT::dynamic_tanh.py::18:20', 'TabDiff::model.py::95:104', 'DiffusionDPO::loss.py::33:39', 'hyla::attention.py::117:120', 'minp::implementation.py::104:106', 'DiffusionDPO::loss.py::57:62', 'llm-sci-use::src/MLE.py::182:188', 'GMFlow::gm_kl_loss.py::20:38', 'fractalgen::models/mar.py::140:148', 'fractalgen::models/mar.py::183:190', 'DyT::dynamic_tanh.py::41:43', 'fractalgen::models/mar.py::358:360', 'DyT::dynamic_tanh.py::31:45', 'hyla::attention.py::356:360', 'fractalgen::models/mar.py::171:177', 'GMFlow::gm_kl_loss.py::26:28', 'llm-sci-use::src/MLE.py::92:96', 'GMFlow::gm_kl_loss.py::22:24', 'fractalgen::models/mar.py::318:321', 'llm-sci-use::src/MLE.py::173:198', 'fractalgen::models/mar.py::328:330', 'DyT::dynamic_tanh.py::35:44', 'minp::implementation.py::114:117', 'TabDiff::model.py::118:120', 'DyT::dynamic_tanh.py::14:16', 'fractalgen::models/mar.py::297:302', 'REPA-E::models/sit.py::217:222', 'DiffusionDPO::loss.py::43:55', 'fractalgen::models/mar.py::351:356', 'minp::implementation.py::96:98', 'DyT::dynamic_tanh.py::37:39', 'llm-sci-use::src/MLE.py::140:144']
        val_instance_ids = ['TabDiff::model.py::85:87', 'llm-sci-use::src/MLE.py::97:101', 'fractalgen::models/mar.py::164:167', 'llm-sci-use::src/MLE.py::127:130', 'llm-sci-use::src/MLE.py::126:156', 'DiffusionDPO::loss.py::49:53', 'hyla::attention.py::345:352', 'llm-sci-use::src/MLE.py::133:151', 'TabDiff::model.py::128:132', 'GMFlow::gm_kl_loss.py::34:36', 'fractalgen::models/mar.py::309:311', 'TabDiff::model.py::79:81', 'DyT::dynamic_tanh.py::12:26', 'llm-sci-use::src/MLE.py::18:32', 'minp::implementation.py::92:94', 'llm-sci-use::src/MLE.py::174:181', 'llm-sci-use::src/MLE.py::189:197', 'DiffusionDPO::loss.py::40:42', 'DiffusionDPO::loss.py::24:68', 'fractalgen::models/mar.py::152:160', 'DyT::dynamic_tanh.py::32:34', 'DiffusionDPO::loss.py::28:30', 'minp::implementation.py::91:118', 'DyT::dynamic_tanh.py::22:24', 'llm-sci-use::src/MLE.py::91:103', 'minp::implementation.py::108:112', 'DiffusionDPO::loss.py::32:63', 'minp::implementation.py::100:102', 'fractalgen::models/mar.py::304:368', 'fractalgen::models/mar.py::345:347', 'llm-sci-use::src/MLE.py::57:65', 'GMFlow::gm_kl_loss.py::30:32', 'llm-sci-use::src/MLE.py::152:155', 'llm-sci-use::src/MLE.py::60:63']
        test_instance_ids = ['Diff-Transformer::model.py::63:67', 'Diff-Transformer::model.py::42:44', 'Diff-Transformer::model.py::101:106', 'Diff-Transformer::model.py::47:52', 'Diff-Transformer::model.py::56:60', 'advantage-alignment::train.py::987:992', 'advantage-alignment::train.py::885:887', 'advantage-alignment::train.py::995:1000', 'advantage-alignment::train.py::883:898', 'advantage-alignment::train.py::884:895', 'Diff-Transformer::model.py::107:109', 'Diff-Transformer::model.py::117:171', 'eomt::eomt.py::22:47', 'eomt::eomt.py::44:46', 'eomt::eomt.py::80:82', 'eomt::eomt.py::34:43', 'eomt::eomt.py::69:74', 'eomt::eomt.py::68:109', 'eomt::eomt.py::31:33', 'eomt::eomt.py::76:78', 'eomt::eomt.py::88:96', 'eomt::eomt.py::98:102', 'eomt::eomt.py::49:55', 'eomt::eomt.py::104:108', 'eomt::eomt.py::113:115', 'eomt::eomt.py::117:119', 'eomt::eomt.py::23:30', 'eomt::eomt.py::128:132', 'eomt::eomt.py::84:86', 'eomt::eomt.py::138:151', 'eomt::eomt.py::140:148', 'eomt::eomt.py::112:135', 'eomt::eomt.py::154:185', 'eomt::eomt.py::189:191', 'eomt::eomt.py::200:203', 'eomt::eomt.py::121:126', 'eomt::eomt.py::207:212', 'eomt::eomt.py::180:182', 'eomt::eomt.py::155:161', 'eomt::eomt.py::163:168', 'eomt::eomt.py::219:223', 'eomt::eomt.py::250:257', 'eomt::eomt.py::193:198', 'eomt::eomt.py::225:248', 'eomt::eomt.py::205:266', 'eomt::eomt.py::188:278', 'eomt::eomt.py::260:265', 'eomt::eomt.py::214:258', 'eomt::eomt.py::268:272', 'GMFlow::lib/models/diffusions/gmflow.py::71:94', 'GMFlow::lib/models/diffusions/gmflow.py::67:69', 'GMFlow::lib/models/diffusions/gmflow.py::100:117', 'GMFlow::lib/models/diffusions/gmflow.py::135:137', 'GPS::model.py::27:31', 'GPS::model.py::23:33', 'GPS::model.py::45:81', 'GMFlow::lib/models/diffusions/gmflow.py::63:65', 'GMFlow::lib/models/diffusions/gmflow.py::441:443', 'GMFlow::lib/models/diffusions/gmflow.py::124:133', 'GMFlow::lib/models/diffusions/gmflow.py::480:497', 'GMFlow::lib/models/diffusions/gmflow.py::456:550', 'GPS::model.py::49:54', 'grid-cell-conformal-isometry::model.py::67:71', 'GPS::model.py::56:64', 'grid-cell-conformal-isometry::model.py::62:65', 'GPS::model.py::66:78', 'grid-cell-conformal-isometry::model.py::82:88', 'grid-cell-conformal-isometry::model.py::53:74', 'grid-cell-conformal-isometry::model.py::77:108', 'GMFlow::lib/models/diffusions/gmflow.py::501:510', 'GMFlow::lib/models/diffusions/gmflow.py::521:538', 'grid-cell-conformal-isometry::model.py::206:209', 'grid-cell-conformal-isometry::model.py::241:244', 'grid-cell-conformal-isometry::model.py::90:93', 'grid-cell-conformal-isometry::model.py::248:252', 'grid-cell-conformal-isometry::model.py::95:105', 'grid-cell-conformal-isometry::model.py::193:199', 'grid-cell-conformal-isometry::model.py::156:159', 'grid-cell-conformal-isometry::model.py::192:212', 'grid-cell-conformal-isometry::model.py::161:164', 'grid-cell-conformal-isometry::model.py::152:167', 'LEN::Synthetic.py::46:50', 'LEN::Synthetic.py::56:60', 'LEN::Synthetic.py::33:51', 'LEN::Synthetic.py::39:44', 'LEN::Synthetic.py::34:37', 'LEN::Synthetic.py::62:68', 'LEN::Synthetic.py::55:69', 'LEN::Synthetic.py::109:119', 'LEN::Synthetic.py::108:165', 'LEN::Synthetic.py::134:136', 'LEN::Synthetic.py::146:157', 'LEN::Synthetic.py::126:131', 'LEN::Synthetic.py::138:144', 'LEN::Synthetic.py::122:164', 'OptimalSteps::OSS.py::21:101', 'OptimalSteps::OSS.py::63:67', 'OptimalSteps::OSS.py::30:35', 'OptimalSteps::OSS.py::71:82', 'OptimalSteps::OSS.py::52:57', 'OptimalSteps::OSS.py::23:37', 'OptimalSteps::OSS.py::77:81', 'OptimalSteps::OSS.py::59:89', 'OptimalSteps::OSS.py::41:100', 'OptimalSteps::OSS.py::91:99', 'schedule_free::adamw_schedulefree_reference.py::143:148', 'schedule_free::adamw_schedulefree_reference.py::154:157', 'schedule_free::adamw_schedulefree_reference.py::150:152', 'schedule_free::adamw_schedulefree_reference.py::131:227', 'OptimalSteps::OSS.py::84:88', 'schedule_free::adamw_schedulefree_reference.py::161:172', 'schedule_free::adamw_schedulefree_reference.py::205:207', 'schedule_free::adamw_schedulefree_reference.py::162:164', 'schedule_free::adamw_schedulefree_reference.py::201:203', 'schedule_free::adamw_schedulefree_reference.py::194:197', 'schedule_free::adamw_schedulefree_reference.py::199:212', 'semanticist::generate.py::8:12', 'schedule_free::adamw_schedulefree_reference.py::209:211', 'schedule_free::adamw_schedulefree_reference.py::214:222', 'schedule_free::adamw_schedulefree_reference.py::219:221', 'REPA-E::train_repae.py::375:379', 'REPA-E::train_repae.py::338:356', 'REPA-E::train_repae.py::396:399', 'semanticist::generate.py::44:51', 'semanticist::generate.py::80:83', 'semanticist::generate.py::53:65', 'semanticist::generate.py::16:22', 'semanticist::generate.py::28:38', 'semanticist::generate.py::67:78', 'schedule_free::adamw_schedulefree_reference.py::215:217', 'REPA-E::train_repae.py::331:336', 'semanticist::generate.py::43:101', 'semanticist::generate.py::92:97', 'SISS::losses.py::86:92', 'SISS::losses.py::42:48', 'semanticist::generate.py::85:90', 'semanticist::generate.py::105:112', 'SISS::losses.py::31:40', 'SISS::losses.py::50:65', 'SISS::losses.py::22:75', 'Tanh-Init::model.py::22:27', 'Tanh-Init::model.py::11:21', 'Tanh-Init::model.py::9:28', 'Tanh-Init::model.py::15:20']
        
        
        all_instance_ids = train_instance_ids + val_instance_ids + test_instance_ids

        rcb = load_tasks(path='gepa_artifact/benchmarks/researchcodebench/researchcodebench.jsonl')
        
        rcb['instance_id'] = rcb.pop('task_id')

        # Process data and remove unnecessary keys
        keys_to_remove = ['gold_snippet', 'unmasked_file']
        if with_gold : 
            keys_to_remove = []

        instances = rcb 
        # Create examples for all relevant instances
        all_examples = []
        for instance in instances:
            if instance['instance_id'] in all_instance_ids:
                for key in keys_to_remove:
                    instance.pop(key, None)

                # Use single input field pattern like other benchmarks
                # ex = dspy.Example(**instance).with_inputs("query",'git_commit','github_repo','instance_id','landmarks','answer')
                ex = dspy.Example(**instance).with_inputs("instance_id","paper_id","snippet_name","masked_file","context_files","paper",)
                all_examples.append(ex)

        # Create splits based on hardcoded instance IDs
        self.train_set = [ex for ex in all_examples if ex.instance_id in train_instance_ids]
        self.val_set = [ex for ex in all_examples if ex.instance_id in val_instance_ids]
        self.test_set = [ex for ex in all_examples if ex.instance_id in test_instance_ids]

        # Set the combined dataset
        self.dataset = self.train_set + self.val_set + self.test_set