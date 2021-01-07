/**
 * Copyright (C) SEI, PKU, PRC. - All Rights Reserved.
 * Unauthorized copying of this file via any medium is
 * strictly prohibited Proprietary and Confidential.
 * Written by Jiajun Jiang<jiajun.jiang@pku.edu.cn>.
 */
package cofix.core.parser.search;

import java.io.*;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

//import mfix.common.util.JavaFiles;
//import mfix.core.node.ast.MethDecl;
import org.eclipse.jdt.core.dom.ASTNode;
import org.eclipse.jdt.core.dom.ASTVisitor;
import org.eclipse.jdt.core.dom.AnonymousClassDeclaration;
import org.eclipse.jdt.core.dom.CompilationUnit;
import org.eclipse.jdt.core.dom.MethodDeclaration;
import org.eclipse.jdt.core.dom.MethodInvocation;
import org.eclipse.jdt.core.dom.SimpleName;
import org.eclipse.jdt.core.dom.Statement;
import org.eclipse.jdt.core.dom.Type;

import cofix.common.config.Constant;
import cofix.common.util.JavaFile;
import cofix.common.util.Pair;
import cofix.core.match.CodeBlockMatcher;
import cofix.core.metric.CondStruct;
import cofix.core.metric.MethodCall;
import cofix.core.metric.OtherStruct;
import cofix.core.metric.Variable;
import cofix.core.parser.NodeUtils;
import cofix.core.parser.ProjectInfo;
import cofix.core.parser.node.CodeBlock;

import static cofix.common.config.Constant.HOME;

/**
 * @author Jiajun
 * @date Jun 29, 2017
 */
public class SimpleFilter {
    private List<CodeBlock> _candidates = new ArrayList<>();
    private CodeBlock _buggyCode = null;
    private Set<Variable> _variables = null;
    private Set<CondStruct.KIND> _condStruct = null;
    private Set<OtherStruct.KIND> _otherStruct = null;
    private Set<String> _methods = null;
    private int _max_line = 0;
    private int DELTA_LINE = 10;


    public SimpleFilter(CodeBlock buggyCode) {
        _buggyCode = buggyCode;
        _variables = new HashSet<>(buggyCode.getVariables());
        _condStruct = new HashSet<>();
        for (CondStruct condStruct : buggyCode.getCondStruct()) {
            _condStruct.add(condStruct.getKind());
        }
        _otherStruct = new HashSet<>();
        for (OtherStruct otherStruct : buggyCode.getOtherStruct()) {
            _otherStruct.add(otherStruct.getKind());
        }
        _methods = new HashSet<>();
        for (MethodCall call : _buggyCode.getMethodCalls()) {
            _methods.add(call.getName());
        }
        _max_line = _buggyCode.getCurrentLine() + DELTA_LINE;
    }

    public List<Pair<CodeBlock, Double>> vectorFilter2(String srcPath, double guard) {
        List<String> files = JavaFile.ergodic(srcPath, new ArrayList<String>());
        List<Pair<CodeBlock, Double>> filtered = new ArrayList<>();
        String method = "public void test(){\n";
        CollectorVisitor collectorVisitor = new CollectorVisitor();

        for (String file : files) {
            CompilationUnit unit;
            try {
                unit = JavaFile.genASTFromFile(file);
            } catch (Exception e) {
                continue;
            }
            collectorVisitor.setUnit(file, unit);
            unit.accept(collectorVisitor);
            filtered = codefilter(filtered);

        }
        if (filtered.size() == 0) {
            return new ArrayList<>();
        }
        List<String> exist = new ArrayList<>();//filter
        List<Pair<CodeBlock, Double>> match = new ArrayList<>();
        for (Pair<CodeBlock, Double> pair : filtered) {
            if (exist.size() > 1000) {
                break;
            }
            exist.add(pair.getFirst().toSrcString().toString());
            match.add(pair);
        }
        for (int i = 0; i < exist.size(); i++) {
            File writejava = new File(HOME + "/java/0/" + i + ".java");
            BufferedWriter writeText = null;
            try {
                writeText = new BufferedWriter(new FileWriter(writejava));
                writeText.write(method + exist.get(i) + "}");
            } catch (IOException e) {
                e.printStackTrace();
            }
            try {
                writeText.flush();
                writeText.close();
            } catch (IOException e) {
                e.printStackTrace();
            }
        }
        File wd = new File("/bin");
        //System.out.println(wd);
        Process proc = null;
        try {
            proc = Runtime.getRuntime().exec("/bin/sh", null, wd);
        } catch (IOException e) {
            e.printStackTrace();
        }
        if (proc != null) {
            BufferedReader in = new BufferedReader(new InputStreamReader(proc.getInputStream()));
            PrintWriter out = new PrintWriter(new BufferedWriter(new OutputStreamWriter(proc.getOutputStream())), true);
            out.println("conda activate treecaps");
            out.println("pwd");
            out.println("/home/cyt/anaconda3/envs/treecaps/bin/python /home/cyt/SimFix/generate_pb.py");
            out.println("/home/cyt/anaconda3/envs/treecaps/bin/python /home/cyt/SimFix/clone/generate_pkl.py");
            out.println("exit");
            //System.out.println("match: "+match.size());

            try {
                String line;
                while ((line = in.readLine()) != null) {
                    System.out.println(line);
                }
                proc.waitFor();
                in.close();
                out.close();
                proc.destroy();
            } catch (Exception e) {
                e.printStackTrace();
            }
        }
        List<Pair<CodeBlock, Double>> sorted = new ArrayList<>();
//        try {
//            BufferedReader in = new BufferedReader(new FileReader(HOME+"/clone/data_simfixion/dict_result.csv"));
//            String str;
//            String[] content = new String[3];
//            str = in.readLine();
//            while ((str = in.readLine()) != null) {
//                int len = str.split(",").length;
//                if(len != 3){
//                    continue;
//                }
//                content = str.split(",");
////			    if(Integer.parseInt(content[1].toString())-2>=match.size()){
////			    	continue;
////				}
//                sorted.add(new Pair<CodeBlock, Double>(match.get(Integer.parseInt(content[1].toString())-2).getFirst(), Double.valueOf(content[2].toString())));
//            }
//            if(match.size()!=sorted.size()){
//                return new ArrayList<>();
//            }
//        } catch (IOException e) {
//            e.printStackTrace();
//        }
        Collections.sort(sorted, new Comparator<Pair<CodeBlock, Double>>() {
            @Override
            public int compare(Pair<CodeBlock, Double> o1, Pair<CodeBlock, Double> o2) {
                if (o1.getSecond() < o2.getSecond()) {
                    return 1;
                } else if (o1.getSecond() > o2.getSecond()) {
                    return -1;
                } else {
                    return 0;
                }
            }
        });
        return sorted;
    }

    public List<Pair<CodeBlock, Double>> vectorFilter(String srcPath, String projectName, String bugId, boolean useSupervised) {
        List<String> files = JavaFile.ergodic(srcPath, new ArrayList<String>());
        String base_url = null;
        String flag = "true";
        if (useSupervised) {
            base_url = HOME + "/TransASTNN/simfix_supervised_data/" + projectName + "/" + bugId;
        } else {
            flag = "false";
            base_url = HOME + "/TransASTNN/simfix_unsupervised_data/" + projectName + "/" + bugId;
        }

        List<Pair<CodeBlock, Double>> filtered = new ArrayList<>();
        String method = "public void test(){\r";
        CollectorVisitor collectorVisitor = new CollectorVisitor();
        File writesrc = new File(base_url + "/src.tsv");
        File writepkl = new File(base_url + "/lables.csv");
        String s1 = "1" + "\t" + "\"" + method + _buggyCode.toSrcString().toString().replace('\n', '\r').replaceAll("\"", "\"\"") + "}" + "\"";
        String s2 = "id1" + "," + "id2" + "," + "label" + "\n";
        BufferedWriter writeTextsrc = null;
        BufferedWriter writeTextpkl = null;
        try {
            writeTextsrc = new BufferedWriter(new FileWriter(writesrc));
            writeTextpkl = new BufferedWriter(new FileWriter(writepkl));
            writeTextpkl.write("id1" + "," + "id2" + "," + "label");
            writeTextsrc.write("1" + "\t" + "\"" + method + _buggyCode.toSrcString().toString().replace('\n', '\r').replaceAll("\"", "\"\"") + "}" + "\"");
        } catch (IOException e) {
            e.printStackTrace();
        }
        for (String file : files) {
            CompilationUnit unit;
            try {
                unit = JavaFile.genASTFromFile(file);
            } catch (Exception e) {
                continue;
            }
            collectorVisitor.setUnit(file, unit);
            unit.accept(collectorVisitor);
            filtered = codefilter(filtered);

        }
        if (filtered.size() == 0) {
            return new ArrayList<>();
        }
        List<String> exist = new ArrayList<>();//filter
        List<Pair<CodeBlock, Double>> match = new ArrayList<>();
        for (Pair<CodeBlock, Double> pair : filtered) {
            if (exist.contains(pair.getFirst().toSrcString().toString()) || pair.getFirst().toSrcString().toString().startsWith("case")) {
                continue;
            }
            if (exist.size() > 1000) {
                break;
            }
            exist.add(pair.getFirst().toSrcString().toString());
            match.add(pair);
        }
        for (int i = 0; i < exist.size(); i++) {
            try {
                writeTextsrc.newLine();
                writeTextpkl.newLine();
                s1 += (i + 2) + "\t" + "\"" + method + exist.get(i).replace('\n', '\r').replaceAll("\"", "\"\"") + "}" + "\"" + "\n";
                s2 += "1" + "," + (i + 2) + "," + "1" + "\n";
                writeTextpkl.write("1" + "," + (i + 2) + "," + "1");
                writeTextsrc.write((i + 2) + "\t" + "\"" + method + exist.get(i).replace('\n', '\r').replaceAll("\"", "\"\"") + "}" + "\"");
                writeTextsrc.flush();
                writeTextpkl.flush();
            } catch (IOException e) {
                e.printStackTrace();
            }
        }
        try {
            writeTextsrc.flush();
            writeTextsrc.close();
            writeTextpkl.flush();
            writeTextpkl.close();
        } catch (IOException e) {
            e.printStackTrace();
        }
        File wd = new File("/bin");
        System.out.println(wd);
        Process proc = null;
        try {
            proc = Runtime.getRuntime().exec("/bin/sh", null, wd);
        } catch (IOException e) {
            e.printStackTrace();
        }
        if (proc != null) {
            BufferedReader in = new BufferedReader(new InputStreamReader(proc.getInputStream()));
            PrintWriter out = new PrintWriter(new BufferedWriter(new OutputStreamWriter(proc.getOutputStream())), true);
            //out.println("conda activate astnn");
            out.println("conda activate genpat");
            //out.println("cd /home/cyt/SimFix/TransASTNN");
            out.println("cd " + HOME + "/TransASTNN");
            out.println("pwd");
            //out.println("/home/cyt/anaconda3/envs/astnn/bin/python /home/cyt/SimFix/TransASTNN/simfix_pipeline.py");
            //out.println("/home/cyt/anaconda3/envs/astnn/bin/python /home/cyt/SimFix/TransASTNN/simfix.py");
            out.println("/root/anaconda3/envs/genpat/bin/python " + HOME + "/TransASTNN/predict_pipeline.py --project_name=" + projectName + " --bug_id=" + bugId + " --predict_baseline=" + flag);
            out.println("/root/anaconda3/envs/genpat/bin/python " + HOME + "/TransASTNN/predict.py --project_name " + projectName + " --bug_id " + bugId + " --predict_baseline " + flag);
            out.println("exit");
            System.out.println("match: " + match.size());

            try {
                String line;
                while ((line = in.readLine()) != null) {
                    System.out.println(line);
                }
                proc.waitFor();
                in.close();
                out.close();
                proc.destroy();
            } catch (Exception e) {
                e.printStackTrace();
            }
        }
        List<Pair<CodeBlock, Double>> sorted = new ArrayList<>();
        try {
            BufferedReader in = new BufferedReader(new FileReader(base_url + "/dict_result.csv"));
            String str;
            String[] content = new String[3];
            str = in.readLine();
            while ((str = in.readLine()) != null) {
                int len = str.split(",").length;
                if (len != 3) {
                    continue;
                }
                content = str.split(",");
//			    if(Integer.parseInt(content[1].toString())-2>=match.size()){
//			    	continue;
//				}
                if (Double.valueOf(content[2]) > 0.97) {
                    sorted.add(new Pair<CodeBlock, Double>(match.get(Integer.parseInt(content[1].toString()) - 2).getFirst(), Double.valueOf(content[2].toString())));
                }
            }
            if (match.size() != sorted.size()) {
                System.out.println("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa");
                System.out.println("=============================================================");
//                return filter(srcPath, 0.3);
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
        Collections.sort(sorted, new Comparator<Pair<CodeBlock, Double>>() {
            @Override
            public int compare(Pair<CodeBlock, Double> o1, Pair<CodeBlock, Double> o2) {
                if (o1.getSecond() < o2.getSecond()) {
                    return 1;
                } else if (o1.getSecond() > o2.getSecond()) {
                    return -1;
                } else {
                    return 0;
                }
            }
        });
        return sorted;
    }

    public List<Pair<CodeBlock, Double>> filter(String srcPath, double guard) {
        List<String> files = JavaFile.ergodic(srcPath, new ArrayList<String>());
        List<Pair<CodeBlock, Double>> filtered = new ArrayList<>();
        CollectorVisitor collectorVisitor = new CollectorVisitor();
        int i = 0;
        for (String file : files) {
//			List<MethDecl> srcNode = JavaFiles.getNodes(file);
//				try {
//					File srcFile = new File("src.txt");
//					srcFile.createNewFile();
//					FileWriter fw = new FileWriter(srcFile.getAbsoluteFile());
//					BufferedWriter bw = new BufferedWriter(fw);
//					for(MethDecl oneNode: srcNode) {
//						bw.write(oneNode.toSrcString().toString());
//						bw.write("\n");
//					}
//					bw.close();
//					String[] cmdarray = {"cd /home/cyt/SimFix",""};
////					Process pro = Runtime.getRuntime().exec(cmdarray);
//					Process pro = Runtime.getRuntime().exec("mv /home/cyt/SimFix/src.txt /home/cyt");
//				} catch (IOException e) {
//					System.err.println("Failed to write file : " + file);
//				}
//			List<ASTNode> myast = new ArrayList<>();
//			for(MethDecl oneNode: srcNode){
//				myast.add(oneNode.getast());
//			}
//			CompilationUnit unit = JavaFiles.genAST(file);
//			CodeBlock buggyblock = new CodeBlock(file,unit, myast);
            CompilationUnit unit;
            try {
                unit = JavaFile.genASTFromFile(file);
            } catch (Exception e) {
                File wrong = new File(file);
                wrong.delete();
                continue;
            }
            collectorVisitor.setUnit(file, unit);
//            if(i<6){
//                unit.accept(collectorVisitor);
//                i++;
//                continue;
//            }
            unit.accept(collectorVisitor);

            filtered = filter(filtered, guard);

        }

        Set<String> exist = new HashSet<>();
        for (Pair<CodeBlock, Double> pair : filtered) {
            if (exist.contains(pair.getFirst().toSrcString().toString())) {
                continue;
            }

            exist.add(pair.getFirst().toSrcString().toString());
            double similarity = CodeBlockMatcher.getRewardSimilarity(_buggyCode, pair.getFirst()) + pair.getSecond();
            pair.setSecond(similarity);
        }

        Collections.sort(filtered, new Comparator<Pair<CodeBlock, Double>>() {
            @Override
            public int compare(Pair<CodeBlock, Double> o1, Pair<CodeBlock, Double> o2) {
                if (o1.getSecond() < o2.getSecond()) {
                    return 1;
                } else if (o1.getSecond() > o2.getSecond()) {
                    return -1;
                } else {
                    return 0;
                }
            }
        });

        return filtered;
    }

    private List<Pair<CodeBlock, Double>> filter(List<Pair<CodeBlock, Double>> filtered, double guard) {
//		List<Pair<CodeBlock, Double>> filtered = new ArrayList<>();
        int delta = Constant.MAX_BLOCK_LINE - _buggyCode.getCurrentLine();
        delta = delta > 0 ? delta : 0;
        guard = guard + ((0.7 - guard) * delta / Constant.MAX_BLOCK_LINE); // 0.9
//		System.out.println("Real guard value : " + guard);
        Set<String> codeRec = new HashSet<>();
        String ss = _buggyCode.toSrcString().toString();
        for (CodeBlock block : _candidates) {
            if (_otherStruct.size() + _condStruct.size() > 0) {
                if ((block.getCondStruct().size() + block.getOtherStruct().size()) == 0) {
                    continue;
                }
            }
            Double similarity = CodeBlockMatcher.getSimilarity(_buggyCode, block);
            if (similarity < guard) {
//				System.out.println("Filtered by similiraty value : " + similarity);
                continue;
            }
//			similarity += CodeBlockMatcher.getRewardSimilarity(_buggyCode, block);
//			if (codeRec.contains(block.toSrcString().toString()) || _buggyCode.hasIntersection(block)) {
//				System.out.println("Duplicate >>>>>>>>>>>>>>>>");
//			} else {
            if (block.getCurrentLine() == 1 && _buggyCode.getCurrentLine() != 1) {
                continue;
            }
            int i = 0;
            boolean hasIntersection = false;
            int replace = -1;
            for (; i < filtered.size(); i++) {
                Pair<CodeBlock, Double> pair = filtered.get(i);
                if (pair.getFirst().hasIntersection(block)) {
                    hasIntersection = true;
                    if (similarity > pair.getSecond()) {
                        replace = i;
                    }
                    break;
                }
            }

            if (hasIntersection) {
                if (replace != -1) {
                    filtered.remove(replace);
                    codeRec.add(block.toSrcString().toString());
                    filtered.add(new Pair<CodeBlock, Double>(block, similarity));
                }
            } else {
                codeRec.add(block.toSrcString().toString());
                filtered.add(new Pair<CodeBlock, Double>(block, similarity));
            }
//			}
        }

        Collections.sort(filtered, new Comparator<Pair<CodeBlock, Double>>() {
            @Override
            public int compare(Pair<CodeBlock, Double> o1, Pair<CodeBlock, Double> o2) {
                if (o1.getSecond() < o2.getSecond()) {
                    return 1;
                } else if (o1.getSecond() > o2.getSecond()) {
                    return -1;
                } else {
                    return 0;
                }
            }
        });
        _candidates = new ArrayList<>();
        if (filtered.size() > 1000) {
            for (int i = filtered.size() - 1; i > 1000; i--) {
                filtered.remove(i);
            }
        }
        return filtered;
    }

    private List<Pair<CodeBlock, Double>> codefilter(List<Pair<CodeBlock, Double>> filtered) {
        for (CodeBlock block : _candidates) {
            if (_otherStruct.size() + _condStruct.size() > 0) {
                if ((block.getCondStruct().size() + block.getOtherStruct().size()) == 0) {
                    continue;
                }
            }
            if (block.getCurrentLine() == 1 && _buggyCode.getCurrentLine() != 1) {
                continue;
            }
//             boolean hasIntersection = false;
//             int replace = -1;
//             for (; i < filtered.size(); i++) {
//                 Pair<CodeBlock, Double> pair = filtered.get(i);
//                 if (pair.getFirst().hasIntersection(block)) {
//                     hasIntersection = true;
//                     replace = i;
//                     break;
//                 }
//             }
            filtered.add(new Pair<CodeBlock, Double>(block, (double) 0));
//             if (hasIntersection) {
//                 if (replace != -1) {
//                     filtered.remove(replace);
//                     filtered.add(new Pair<CodeBlock, Double>(block, (double) 0));
//                 }
//             } else {
//                 filtered.add(new Pair<CodeBlock, Double>(block, (double) 0));
//             }
//			}
        }
        _candidates = new ArrayList<>();
        if (filtered.size() > 1000) {
            for (int i = filtered.size() - 1; i > 1000; i--) {
                filtered.remove(i);
            }
        }
        return filtered;
    }

    class CollectorVisitor extends ASTVisitor {

        private CompilationUnit _unit = null;
        private String _fileName = null;

        public void setUnit(String fileName, CompilationUnit unit) {
            _fileName = fileName;
            _unit = unit;
        }

        @Override
        public boolean visit(SimpleName node) {
            String name = node.getFullyQualifiedName();
            if (Character.isUpperCase(name.charAt(0))) {
                return true;
            }
            Pair<String, String> classAndMethodName = NodeUtils.getTypeDecAndMethodDec(node);
            Type type = ProjectInfo.getVariableType(classAndMethodName.getFirst(), classAndMethodName.getSecond(), name);
            Variable variable = new Variable(null, name, type);
            boolean match = false;
            if (_variables.contains(variable) || _methods.contains(name) || sameStructure(node)) {
                match = true;
            } else {
                ASTNode parent = node.getParent();
                while (parent != null && !(parent instanceof Statement)) {
                    if (parent instanceof MethodInvocation) {
                        if (_methods.contains(((MethodInvocation) parent).getName().getFullyQualifiedName())) {
                            match = true;
                        }
                        break;
                    }
                    parent = parent.getParent();
                }
            }
            if (match) {
                ASTNode parent = node.getParent();
                Statement statement = null;
                while (parent != null && !(parent instanceof MethodDeclaration)) {
                    parent = parent.getParent();
                    if (statement == null && parent instanceof Statement) {
                        statement = (Statement) parent;
                    }
                }
                // filter out anonymous classes
                if (parent != null && !(parent.getParent() instanceof AnonymousClassDeclaration)) {
                    int line = _unit.getLineNumber(node.getStartPosition());
                    CodeSearch codeSearch = new CodeSearch(_unit, line, _buggyCode.getCurrentLine(), statement);
                    CodeBlock codeBlock = new CodeBlock(_fileName, _unit, codeSearch.getASTNodes());
//					CodeBlock codeBlock = new CodeBlock(_fileName, _unit, codeSearch.getASTNodes());
                    if (codeBlock.getCurrentLine() < _max_line) {
                        _candidates.add(codeBlock);
                    }
                }
            }
            return true;
        }

        private boolean sameStructure(SimpleName name) {
            return false;
//			if(_condStruct.size() == 0 && _otherStruct.size() == 0){
//				return false;
//			}
//			ASTNode parent = name.getParent();
//			Object kind = null;
//			while(parent != null){
//				if(parent instanceof MethodDeclaration){
//					break;
//				} else if(parent instanceof IfStatement){
//					kind = CondStruct.KIND.IF;
//					break;
//				} else if(parent instanceof SwitchStatement){
//					kind = CondStruct.KIND.SC;
//					break;
//				} else if(parent instanceof ReturnStatement){
//					kind = OtherStruct.KIND.RETURN;
//					break;
//				} else if(parent instanceof ConditionalExpression){
//					kind = CondStruct.KIND.CE;
//					break;
//				} else if(parent instanceof ThrowStatement){
//					kind = OtherStruct.KIND.THROW;
//					break;
//				}
//				parent = parent.getParent();
//			}
//			if(kind == null){
//				return false;
//			}
//			if(_condStruct.contains(kind) || _otherStruct.contains(kind)){
//				return true;
//			}
//			return false;
        }
    }

}